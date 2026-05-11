# NOTE: Code adapted from LSMI estimator - https://github.com/GeWu-Lab/LSMI_Estimator

from utils import *
from entropy_estimator import MargKernel
import torch
import hydra
import os


def _resolve_log_class_priors(cfg=None, provided_priors=None):
    """Return log p(y) on cfg.device.

    Priority:
    1. ``provided_priors`` argument, if not None.
    2. ``cfg.class_priors``, if set.
    3. Uniform prior ``p(y) = 1/n_classes`` (default — reproducible regardless of
       label distribution; matches the legacy LSMI estimator's assumption).

    Pass ``cfg.class_priors`` (or ``provided_priors``) to use empirical /
    non-uniform priors explicitly.
    """
    if cfg is None:
        raise ValueError("cfg is required to resolve class priors for MI estimation.")

    n_classes = int(cfg.n_classes)
    if provided_priors is None and hasattr(cfg, 'class_priors') and cfg.class_priors is not None:
        provided_priors = cfg.class_priors

    if provided_priors is None:
        priors = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float32, device=cfg.device)
    else:
        priors = torch.as_tensor(provided_priors, dtype=torch.float32, device=cfg.device)
        if priors.numel() != n_classes:
            raise ValueError(f"class_priors length ({priors.numel()}) must match n_classes ({n_classes}).")
        priors = priors / priors.sum().clamp_min(1e-12)

    return priors.clamp_min(1e-12).log()


def save_trained_estimators(cfg, discriminator, entropy_estimator, output_dir=None, tag='latest'):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'saved_estimators')
    os.makedirs(output_dir, exist_ok=True)

    state_path = os.path.join(output_dir, f'mi_estimators_{tag}_state_dict.pt')
    full_model_path = os.path.join(output_dir, f'mi_estimators_{tag}_full_models.pt')

    checkpoint = {
        'meta': {
            'input_size_1': int(cfg.input_size_1),
            'input_size_2': int(cfg.input_size_2),
            'embed_size': int(cfg.embed_size),
            'n_classes': int(cfg.n_classes),
        },
        'discriminator_state_dicts': [m.state_dict() for m in discriminator],
        'entropy_state_dicts': [m.state_dict() for m in entropy_estimator],
    }
    torch.save(checkpoint, state_path)

    # Optional raw model serialization for quick reloads in same codebase environment.
    if getattr(cfg, 'save_full_models', True):
        torch.save(
            {
                'meta': checkpoint['meta'],
                'discriminator_models': discriminator,
                'entropy_models': entropy_estimator,
            },
            full_model_path,
        )
        print(f"Saved MI estimator full models to: {full_model_path}")

    print(f"Saved MI estimator state_dict checkpoint to: {state_path}") # dictionary of model parameters. saved location in train.yaml config file.
    return state_path


def _build_discriminator_models(cfg, device=None):
    use_device = device if device is not None else cfg.device
    model_1 = cls_network(input_dim=cfg.input_size_1, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(use_device)
    model_2 = cls_network(input_dim=cfg.input_size_2, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(use_device)
    model_j = cls_network(input_dim=cfg.input_size_1 + cfg.input_size_2, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(use_device)
    return [model_1, model_2, model_j]


def _build_entropy_models(cfg, device=None):
    use_device = device if device is not None else cfg.device
    model_1 = MargKernel(dim=cfg.input_size_1).to(use_device)
    model_2 = MargKernel(dim=cfg.input_size_2).to(use_device)
    return [model_1, model_2]


def load_trained_estimators(cfg, checkpoint_path, device=None):
    use_device = device if device is not None else cfg.device
    checkpoint = torch.load(checkpoint_path, map_location=use_device)
    meta = checkpoint.get('meta', {})

    if 'input_size_1' in meta:
        cfg.input_size_1 = int(meta['input_size_1'])
    if 'input_size_2' in meta:
        cfg.input_size_2 = int(meta['input_size_2'])
    if 'embed_size' in meta:
        cfg.embed_size = int(meta['embed_size'])
    if 'n_classes' in meta:
        cfg.n_classes = int(meta['n_classes'])

    discriminator = _build_discriminator_models(cfg, device=use_device) # a list of 3 models: modality 1, modality 2, joint
    entropy_estimator = _build_entropy_models(cfg, device=use_device) # a list of 2 models: modality 1, modality 2

    for model, state in zip(discriminator, checkpoint['discriminator_state_dicts']):
        model.load_state_dict(state)
        model.eval()

    for model, state in zip(entropy_estimator, checkpoint['entropy_state_dicts']):
        model.load_state_dict(state)
        model.eval()

    print(f"Loaded MI estimators from: {checkpoint_path}")
    return discriminator, entropy_estimator

def RUS_adjustment(rus):
    """
    Adjusts the input tensors (r, u1, u2, s) while preserving certain sums
    and the original device of the tensors. The adjustment aims to make the
    means of these components non-negative based on a specific priority:

    1. If the mean of 'r' (R_mean) or 's' (S_mean) is negative, an adjustment
       factor is calculated to make both R_mean and S_mean non-negative.
       This adjustment might consequently alter the means of 'u1' (U1_mean)
       and 'u2' (U2_mean), potentially making them negative.

    2. If R_mean and S_mean are already non-negative, but U1_mean or U2_mean
       is negative, the adjustment factor is calculated to make both U1_mean
       and U2_mean non-negative. This adjustment might, in turn, make
       R_mean or S_mean negative if they were small positive values.

    The adjustment maintains the following sum properties for the means:
    - (R_mean + U1_mean + U2_mean + S_mean) remains unchanged.
    - (R_mean + U1_mean) remains unchanged.
    - (R_mean + U2_mean) remains unchanged.

    Args:
        rus (tuple or list): A collection of four PyTorch tensors (r, u1, u2, s).

    Returns:
        tuple: A tuple of four adjusted PyTorch tensors (r_adjusted, u1_adjusted,
               u2_adjusted, s_adjusted), on the same device as the input tensors.
    """
    r_orig, u_1_orig, u_2_orig, s_orig = rus

    R_mean = r_orig.detach().mean()
    U1_mean = u_1_orig.detach().mean()
    U2_mean = u_2_orig.detach().mean()
    S_mean = s_orig.detach().mean()

    adj_factor = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)

    # Priority 1: Address negative mean of r or s
    if R_mean < 0 or S_mean < 0:
        adj_factor = -torch.min(R_mean, S_mean)
          
    # Priority 2: If means of r and s are non-negative, address negative mean of u1 or u2
    elif U1_mean < 0 or U2_mean < 0:
        adj_factor = torch.min(U1_mean, U2_mean)

    r_adjusted = r_orig + adj_factor
    u_1_adjusted = u_1_orig - adj_factor
    u_2_adjusted = u_2_orig - adj_factor
    s_adjusted = s_orig + adj_factor
    
    return r_adjusted, u_1_adjusted, u_2_adjusted, s_adjusted


def get_entropy(dataloader, model, modality = 'modality_1', cfg = None):
    model.eval()
    info = []
    with torch.no_grad():
        losses = 0.0
        for batch in dataloader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device = cfg.device) # obtain_feature_input from utils.py
            if modality == "modality_1":
                input_data = modal_1
            elif modality == "modality_2":
                input_data = modal_2
            batch_size = input_data.shape[0]
            loss = model(input_data)
            info.append(loss)
            losses = losses + torch.mean(loss).item() * batch_size
    info = torch.cat(info, dim = 0).detach()
    return info


def get_mutual_info(dataloader, model, modality = 'modality_1', cfg = None, log_class_priors = None):
    model.eval()
    info = []
    if log_class_priors is None: # for log p(y) if there are non-uniform classes (aka unbalanced labels).
        log_class_priors = _resolve_log_class_priors(cfg=cfg)
    with torch.no_grad():
        infos = 0.0
        for batch in dataloader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device = cfg.device)
            if modality == "modality_1":    
                input_data = modal_1
            elif modality == "modality_2":
                input_data = modal_2
            elif modality == "modality_12":
                input_data = torch.cat([modal_1, modal_2], dim = 1)
            batch_size = input_data.shape[0]
            rows = torch.arange(batch_size, device=input_data.device)
            out = model(input_data)
            info_cur = torch.nn.functional.log_softmax(out, dim=1)[rows, labels] - log_class_priors[labels] # i(x;y)=log p(y|x)-log p(y)
            info.append(info_cur)
            infos = infos + torch.mean(info_cur).item() * batch_size
    info = torch.cat(info, dim = 0).detach()
    return info
 
                          
def LSMI_estimation(dataloader, discriminator, entropy_estimator, cfg = None, return_per_sample = False):
    """LSMI decomposition over a dataloader.

    Args:
        return_per_sample: when True, additionally return per-sample tensors
            (r, u_1, u_2, s) aligned with the dataloader's iteration order.
    """
    log_class_priors = _resolve_log_class_priors(cfg=cfg) # compute log p(y) before passing it to compute the pointwise information terms.
    I_X1Y = get_mutual_info(dataloader, discriminator[0], modality = 'modality_1', cfg = cfg, log_class_priors = log_class_priors) # i(x;y)=log p(y|x)-log p(y) with class-prior correction.
    I_X2Y = get_mutual_info(dataloader, discriminator[1], modality = 'modality_2', cfg = cfg, log_class_priors = log_class_priors)
    I_X1X2Y = get_mutual_info(dataloader, discriminator[2], modality = 'modality_12', cfg = cfg, log_class_priors = log_class_priors)
    H_X1 = get_entropy(dataloader, entropy_estimator[0], modality = 'modality_1', cfg = cfg)
    H_X2 = get_entropy(dataloader, entropy_estimator[1], modality = 'modality_2', cfg = cfg)

    r_plus = torch.minimum(H_X1, H_X2)
    r_minus = torch.minimum(H_X1 - I_X1Y, H_X2 - I_X2Y)
    r = r_plus - r_minus

    r =  r_plus - r_minus
    u_1 = I_X1Y - r
    u_2 = I_X2Y - r
    s = I_X1X2Y - r - u_1 - u_2
    # print(f"I_X1Y: {torch.mean(I_X1Y)}, I_X2Y: {torch.mean(I_X2Y)}, I_X1X2Y: {torch.mean(I_X1X2Y)}")
    # print(f"before adjustment r: {torch.mean(r)}, u_1: {torch.mean(u_1)}, u_2: {torch.mean(u_2)}, s: {torch.mean(s)}")
    r, u_1, u_2, s = RUS_adjustment([r, u_1, u_2, s])

    R_minus = torch.mean(r_minus)
    R_plus = torch.mean(r_plus)

    R = torch.mean(r)
    U_1 = torch.mean(u_1)
    U_2 = torch.mean(u_2)
    S = torch.mean(s)

    # print(f"R_minus: {R_minus.item():.4f}, R_plus: {R_plus.item():.4f}")
    print(f"R: {R.item():.4f}, U1: {U_1.item():.4f}, U2: {U_2.item():.4f}, S: {S.item():.4f}")
    if return_per_sample:
        return R, U_1, U_2, S, {'r': r.detach().cpu(), 'u1': u_1.detach().cpu(), 'u2': u_2.detach().cpu(), 's': s.detach().cpu()}
    return R, U_1, U_2, S
          

def obtain_discriminator(cfg, train_loader, val_loader):
    print("Training discriminators")
    models = _build_discriminator_models(cfg) # modality 1: image, modality 2: text, joint: both
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = cfg.num_epochs_discriminator

    # Early stopping defaults
    early_stopping = getattr(cfg, 'early_stopping', True)
    patience = int(getattr(cfg, 'es_patience', 5))
    min_delta = float(getattr(cfg, 'es_min_delta', 0.0))
    best_val = float('inf')
    best_state = [
        {k: v.clone().detach().cpu() for k, v in model.state_dict().items()} for model in models
    ]
    bad_epochs = 0

    for epoch in range(num_epochs):
        for m in models:
            m.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device=cfg.device)
            batch_size = modal_1.shape[0]
            out_1 = models[0](modal_1)
            out_2 = models[1](modal_2)
            out_j = models[2](torch.cat([modal_1, modal_2], dim=1))
            optimizer.zero_grad()
            loss = (
                criterion(out_1, labels)
                + criterion(out_2, labels)
                + criterion(out_j, labels)
            )
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch_size
            train_count += batch_size

        scheduler.step()

        # Validation
        val_loss = _eval_discriminator_loss(models, val_loader, cfg)
        train_loss = train_total / max(1, train_count)

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_state = [
                {k: v.clone().detach().cpu() for k, v in model.state_dict().items()} for model in models
            ]
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, best_val: {best_val:.4f}')

        if early_stopping and bad_epochs >= patience:
            print(f"Early stopping (disc) at epoch {epoch + 1} with best val {best_val:.4f}")
            break

    # Load best weights
    for model, state in zip(models, best_state):
        model.load_state_dict(state)

    return models


def _eval_discriminator_loss(models, loader, cfg):
    criterion = torch.nn.CrossEntropyLoss()
    for m in models:
        m.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device=cfg.device)
            batch_size = modal_1.shape[0]
            out_1 = models[0](modal_1)
            out_2 = models[1](modal_2)
            out_j = models[2](torch.cat([modal_1, modal_2], dim=1))
            loss = (
                criterion(out_1, labels)
                + criterion(out_2, labels)
                + criterion(out_j, labels)
            )
            total += loss.item() * batch_size
            count += batch_size
    return total / max(1, count)


def obtain_entropy_estimator(cfg, train_loader, val_loader):
    print("Training entropy estimators")
    models = _build_entropy_models(cfg)
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    num_epochs = cfg.num_epochs_entropy_estimator

    # Early stopping defaults
    early_stopping = getattr(cfg, 'early_stopping', True)
    patience = int(getattr(cfg, 'es_patience', 5))
    min_delta = float(getattr(cfg, 'es_min_delta', 0.0))
    best_val = float('inf')
    best_state = [
        {k: v.clone().detach().cpu() for k, v in model.state_dict().items()} for model in models
    ]
    bad_epochs = 0

    for epoch in range(num_epochs):
        for m in models:
            m.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device=cfg.device)
            batch_size = modal_1.shape[0]
            optimizer.zero_grad()
            loss = models[0](modal_1) + models[1](modal_2)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * batch_size
            train_count += batch_size

        scheduler.step()

        # Validation
        val_loss = _eval_entropy_loss(models, val_loader, cfg)
        train_loss = train_total / max(1, train_count)

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_state = [
                {k: v.clone().detach().cpu() for k, v in model.state_dict().items()} for model in models
            ]
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, best_val: {best_val:.4f}')

        if early_stopping and bad_epochs >= patience:
            print(f"Early stopping (entropy) at epoch {epoch + 1} with best val {best_val:.4f}")
            break

    # Load best weights
    for model, state in zip(models, best_state):
        model.load_state_dict(state)

    return models


def _eval_entropy_loss(models, loader, cfg):
    for m in models:
        m.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device=cfg.device)
            batch_size = modal_1.shape[0]
            loss1 = models[0](modal_1)
            loss2 = models[1](modal_2)
            # In eval mode, MargKernel returns per-sample losses; take mean
            if loss1.dim() > 0:
                loss1 = loss1.mean()
            if loss2.dim() > 0:
                loss2 = loss2.mean()
            loss = loss1 + loss2
            total += float(loss.item()) * batch_size
            count += batch_size
    return total / max(1, count)


def estimation_main(cfg, feature_dir = None):
    loaders = get_loader(cfg, feature_dir)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = None
    discriminator = obtain_discriminator(cfg, train_loader=train_loader, val_loader=val_loader)
    entropy_estimator = obtain_entropy_estimator(cfg, train_loader=train_loader, val_loader=val_loader)

    save_dir = getattr(cfg, 'estimator_save_dir', os.path.join(os.path.dirname(__file__), 'saved_estimators'))
    save_tag = getattr(cfg, 'estimator_save_tag', 'latest')
    save_trained_estimators(cfg, discriminator, entropy_estimator, output_dir=save_dir, tag=save_tag) # save the trained model with the best val loss.

    print("[Eval] Train split:")
    LSMI_estimation(train_loader, discriminator, entropy_estimator, cfg)
    print("[Eval] Val split:")
    LSMI_estimation(val_loader, discriminator, entropy_estimator, cfg)
    if test_loader is not None:
        print("[Eval] Test split:")
        LSMI_estimation(test_loader, discriminator, entropy_estimator, cfg)
    # LSMI_estimation(test_loader, discriminator, entropy_estimator, cfg)


@hydra.main(config_path='.', config_name='train', version_base=None)
def main(cfg):
    setup_seed(cfg.random_seed)
    data_path = cfg.data_path  # NOTE: configure the path to the features (.pt file) in the train.yaml config file.
    infer_input_dims_from_pt(data_path, cfg) # infer_input_dims_from_pt from utils.py
    estimation_main(cfg, data_path)

if __name__ == '__main__':
	main()