import numpy as np
from PIL import Image
import torch


def pinhole_to_sphere_coordinates(image_path, h_fov_deg=70.0, dx=0.0):
    """
    Projicerer hvert pixel i et pinhole-kamera billede til 3D koordinater på enhedskuglen.
    
    Args:
        image_path: Sti til billedet
        h_fov_deg: Horizontal field of view i grader
    
    Returns:
        xyz_coords: numpy array med shape (H, W, 3) hvor xyz_coords[y, x] = [x, y, z] på enhedskuglen
        mask: boolean array (H, W) der viser hvilke pixels der er gyldige (True = gyldig projektion)
        img: det indlæste billede som numpy array
    """
    
    # Indlæs billede
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    
    # Beregn field of view
    h_fov = np.radians(h_fov_deg)
    # Vertikal FOV baseret på billedets aspect ratio
    v_fov = 2.0 * np.arctan(np.tan(h_fov / 2.0) * (H / W))
    
    # Lav et grid af pixel-koordinater
    # u = horisontale pixel-koordinater [0, W-1]
    # v = vertikale pixel-koordinater [0, H-1]
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    # Normaliser pixel-koordinater til [-1, 1]
    # Centrum af billedet er (0, 0)
    u_norm = (uu - (W - 1) / 2.0) / (W / 2.0)  # -1 til 1
    v_norm = (vv - (H - 1) / 2.0) / (H / 2.0)  # -1 til 1
    
    # Konverter normaliserede koordinater til vinkler
    # u_norm i [-1, 1] svarer til azimuth i [-h_fov/2, h_fov/2]
    # v_norm i [-1, 1] svarer til elevation i [-v_fov/2, v_fov/2]
    # (negativ fordi y-aksen i billeder går nedad)
    azimuth = u_norm * (h_fov / 2.0)      # horisontale vinkel
    elevation = -v_norm * (v_fov / 2.0)   # vertikale vinkel (negativ pga. image coords)
    
    # Konverter sfæriske koordinater til 3D kartesiske koordinater på enhedskuglen
    # Antag at kameraet kigger ned ad z-aksen
    # azimuth = 0, elevation = 0 peger lige frem (z = 1)
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.cos(azimuth)
    
    # Alle punkter ligger på enhedskuglen per konstruktion
    # (fordi vi bruger cos/sin)
    # Men vi laver en mask, der viser hvilke pixels der er inden for FOV
    mask = np.ones((H, W), dtype=bool)
    
    # Stack til (H, W, 3) array
    xyz_coords = np.stack([x, y, z], axis=-1)
    
    return xyz_coords, mask, img


def R2toR3(C, P0, eps=1e-12):
    x, y = C
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape; got {x.shape} and {y.shape}")

    P0 = np.asarray(P0, dtype=float)
    if P0.shape != (3,):
        raise ValueError("P0 must be length-3 (x0,y0,z0)")

    x0, y0, z0 = P0

    w = np.array([y0, -x0, 0.0], dtype=float)
    nw = np.linalg.norm(w)
    if nw < eps:
        raise ValueError("constructed w vector is (near) zero; choose a different P0")
    
    i = w / nw

    v = np.array([-x0 * z0, -y0 * z0, x0**2 + y0**2], dtype=float)
    # v = v - np.dot(v, i) * i
    print("unortogonal part:", np.dot(v, i) * i)
    nv = np.linalg.norm(v)
    if nv < eps:
        raise ValueError("constructed v vector is (near) zero after orthogonalization; cannot form basis")

    j = v / nv

    mapped = P0 + x[..., None] * i + y[..., None] * j

    return mapped


# ---------- R2 -> R3 (PyTorch) ----------
def R2toR3_torch(C, P0, eps=1e-12):
    """
    C: tensor (..., 2) coordinates (x,y)
    P0: tensor (3,) camera/image center vector
    returns: mapped points shape (..., 3)
    """
    x = C[..., 0]
    y = C[..., 1]
    # Ensure shapes
    P0 = P0.reshape(3)
    x0, y0, z0 = P0[0], P0[1], P0[2]

    # build i
    # zero = torch.zeros(1, dtype=P0.dtype, device=P0.device)[0]
    # w = torch.stack([y0, -x0, zero])   # w har samme dtype/device/grad som y0,x0
    # w = torch.tensor([y0, -x0, 0.0], dtype=P0.dtype, device=P0.device)
    w = torch.stack([y0, -x0, torch.zeros_like(x0)])

    nw = torch.norm(w)
    if nw.item() < eps:
        raise ValueError("w near zero; choose different P0")
    i = w / nw

    # build v then orthogonalize against i
    # v = torch.tensor([-x0 * z0, -y0 * z0, x0**2 + y0**2], dtype=P0.dtype, device=P0.device)
    v = torch.stack([-x0 * z0, -y0 * z0, x0**2 + y0**2])
    
    v = v - torch.dot(v, i) * i   # <-- orthogonalize
    nv = torch.norm(v)
    if nv.item() < eps:
        raise ValueError("v near zero after orthogonalization")
    j = v / nv

    # combine: mapped = P0 + x * i + y * j
    # shape handling: x,y can be (...), so we expand dims for broadcast
    mapped = P0 + x.unsqueeze(-1) * i + y.unsqueeze(-1) * j
    return mapped  # (..., 3)


def little_loss(P, Q, C_P, C_Q, k):
    map_P = R2toR3(C_P, P)
    map_Q = R2toR3(C_Q, Q)
    norm_map_P = np.linalg.norm(map_P)
    norm_map_Q = np.linalg.norm(map_Q)
    loss = k * np.arccos(
        np.dot(map_P, map_Q) / (norm_map_P * norm_map_Q)
    ) ** 2
    return loss


# ---------- Total loss over all pairs ----------
def total_loss(P, Q, C_P, C_Q, K):
    """
    P, Q: torch tensors shape (3,) - parameters to optimize
    C_P: tensor (M,2)
    C_Q: tensor (N,2)
    K: tensor (M,N) weight matrix for pairwise losses
    returns: scalar loss
    """
    # map all points
    M_pts = R2toR3_torch(C_P, P)  # (M,3)
    N_pts = R2toR3_torch(C_Q, Q)  # (N,3)

    # normalize
    M_norm = torch.norm(M_pts, dim=1, keepdim=True)  # (M,1)
    N_norm = torch.norm(N_pts, dim=1, keepdim=True)  # (N,1)

    # compute pairwise dot and cross norms via broadcasting
    # expand to (M,N,3)
    A = M_pts.unsqueeze(1)  # (M,1,3)
    B = N_pts.unsqueeze(0)  # (1,N,3)

    dot = (A * B).sum(dim=2)                     # (M,N)
    cross = torch.cross(A.expand(-1, B.shape[1], -1), B.expand(A.shape[0], -1, -1), dim=2)
    cross_norm = torch.norm(cross, dim=2)        # (M,N)

    # angle via atan2
    angle = torch.atan2(cross_norm, dot)         # (M,N)

    # squared angle weighted by K
    loss_matrix = K * (angle ** 2)
    return loss_matrix.sum()


# ---------- Example training loop ----------
def optimize_PQ(C_P_np, C_Q_np, K_np, P_init=None, Q_init=None, lr=1e-2, steps=2000, device='cpu'):
    dtype = torch.float64
    device = torch.device(device)

    C_P = torch.tensor(C_P_np, dtype=dtype, device=device)
    C_Q = torch.tensor(C_Q_np, dtype=dtype, device=device)
    K = torch.tensor(K_np, dtype=dtype, device=device)

    if P_init is None:
        P = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        P = torch.tensor(P_init, dtype=dtype, device=device)
    if Q_init is None:
        Q = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        Q = torch.tensor(Q_init, dtype=dtype, device=device)

    P = torch.nn.Parameter(P)
    Q = torch.nn.Parameter(Q)

    opt = torch.optim.Adam([P, Q], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        L = total_loss(P, Q, C_P, C_Q, K)
        L.backward()
        opt.step()

        if step % 200 == 0 or step == steps-1:
            print(f"step {step:4d} loss {L.item():.6e} P [{', '.join(f'{x:.4f}' for x in P.data.tolist())}] Q [{', '.join(f'{x:.4f}' for x in Q.data.tolist())}]")
            print()

    return P.detach().cpu().numpy(), Q.detach().cpu().numpy()


def optimize_PQ_with_visualization(C_P_np, C_Q_np, K_np, colors_P, colors_Q, 
                                    P_init=None, Q_init=None, 
                                    lr=1e-2, steps=2000, viz_every=100, device='cpu'):
    """
    Optimering med løbende visualisering af overlap
    """
    import matplotlib.pyplot as plt
    
    dtype = torch.float64
    device = torch.device(device)
    
    C_P = torch.tensor(C_P_np, dtype=dtype, device=device)
    C_Q = torch.tensor(C_Q_np, dtype=dtype, device=device)
    K = torch.tensor(K_np, dtype=dtype, device=device)
    
    if P_init is None:
        P = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        P = torch.tensor(P_init, dtype=dtype, device=device)
    
    if Q_init is None:
        Q = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        Q = torch.tensor(Q_init, dtype=dtype, device=device)
    
    P = torch.nn.Parameter(P)
    Q = torch.nn.Parameter(Q)
    opt = torch.optim.Adam([P, Q], lr=lr)
    
    # Setup plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    losses = []
    
    for step in range(steps):
        opt.zero_grad()
        L = total_loss(P, Q, C_P, C_Q, K)
        L.backward()
        opt.step()
        
        losses.append(L.item())
        
        if step % 10 == 0:
            print(f"step {step:4d} loss {L.item():.6e} P [{', '.join(f'{x:.4f}' for x in P.data.tolist())}] Q [{', '.join(f'{x:.4f}' for x in Q.data.tolist())}]")
        
        if step % viz_every == 0 or step == steps - 1:
            M_pts = R2toR3_torch(C_P, P).detach().cpu().numpy()
            N_pts = R2toR3_torch(C_Q, Q).detach().cpu().numpy()
            
            # Project to 2D spherical coords
            def to_angles(xyz):
                x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                azimuth = np.arctan2(x, z)
                r = np.sqrt(x**2 + y**2 + z**2)
                elevation = np.arcsin(np.clip(y / (r + 1e-10), -1, 1))
                return np.degrees(azimuth), np.degrees(elevation)
            
            az_P, el_P = to_angles(M_pts)
            az_Q, el_Q = to_angles(N_pts)
            
            ax1.clear()
            ax2.clear()
            
            ax1.scatter(az_P, el_P, c=colors_P/255.0, s=10, alpha=0.7, 
                       edgecolors='blue', linewidths=0.3, label='Image P')
            ax1.scatter(az_Q, el_Q, c=colors_Q/255.0, s=10, alpha=0.7,
                       edgecolors='red', linewidths=0.3, label='Image Q')
            ax1.set_xlabel('Azimuth (°)')
            ax1.set_ylabel('Elevation (°)')
            ax1.set_title(f'Step {step} - Loss: {L.item():.3e}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(losses, 'b-', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss Convergence')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
    
    plt.ioff()
    plt.show()
    
    return P.detach().cpu().numpy(), Q.detach().cpu().numpy()