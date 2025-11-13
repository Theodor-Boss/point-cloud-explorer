import argparse
# from .simulation.factory import create_grid
# from .visualize import plot_grid
from .mapping import pinhole_to_sphere_coordinates, R2toR3, R2toR3_torch, optimize_PQ, optimize_PQ_with_visualization
from .visualize import visualize_multiple_sphere_projection, simple_visualize_plane_projection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from PIL import Image
import torch
# from my_module.viewer import FirstPersonViewer
from .python_export import export_pointcloud_to_json, export_lines_to_json


def k_value():
    pass


def run() -> None:
    parser = argparse.ArgumentParser(description="Run grid simulation (simple).")
    parser.add_argument("--impl", choices=["dict", "kdtree"], default="dict",
                        help="Which grid implementaion to use")
    parser.add_argument("--width", type=int, default=100, help="Grid width")
    parser.add_argument("--height", type=int, default=100, help="Grid height")
    parser.add_argument("--entities", type=int, default=100, help="Number of random entities to add")
    parser.add_argument("--plot", action="store_true", help="Show a plot of the grid after initialization")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible random fill")
    args = parser.parse_args()


    # C = (2, 1)

    # xs = np.linspace(-1, 1, 10)
    # ys = np.linspace(-1, 1, 10)
    # X, Y = np.meshgrid(xs, ys, indexing='xy')
    # # print(X.shape)

    # # P = (-2.14, 0.89, 0.75)
    # P = (2.0, 2.0, 2.0)

    # points3 = R2toR3((X, Y), P)

    # # print(points3)
    # # print(points3[0,0])

    # simple_visualize_plane_projection(points3)


    # Indlæs billede
    image_path_P = "my_module/frame1.jpg"
    image_path_Q = "my_module/frame2.jpg"
    img_P = np.array(Image.open(image_path_P).convert("RGB"))
    img_Q = np.array(Image.open(image_path_Q).convert("RGB"))
    # Hoo, Woo = img.shape[:2]

    # print(img)
    # print("O")
    # print(Woo)

    # print(img.shape)

    H_P, W_P = img_P.shape[:2]  # 540, 960  # (540, 960)
    H_Q, W_Q = img_Q.shape[:2]  # 540, 960  # (540, 960)

    s = 20

    y_P, x_P = np.mgrid[0:H_P:s, 0:W_P:s]   # giver to arrays med form (H, W)
    y_Q, x_Q = np.mgrid[0:H_Q:s, 0:W_Q:s]   # giver to arrays med form (H, W)
    
    C_P_np = np.stack([x_P, y_P], axis=-1)  # shape: (H, W, 2)
    C_Q_np = np.stack([x_Q, y_Q], axis=-1)  # shape: (H, W, 2)

    print(f"Grid shapes: C_P={C_P_np.shape}, C_Q={C_Q_np.shape}")

    # METODE 1: Brug gennemsnit af patch omkring hver pixel (mere robust!)
    patch_size = s // 2  # halv størrelse af sampling afstand
    

    def get_patch_colors(img, coords_grid, patch_size):
        """Beregn gennemsnitlige farver for patches omkring hver koordinat"""
        h, w = coords_grid.shape[:2]
        colors = np.zeros((h, w, 3))
        
        for i in range(h):
            for j in range(w):
                x, y = coords_grid[i, j]

                if patch_size == 0:
                    patch = img[y, x]
                    colors[i, j] = patch
                else:
                    # Definer patch bounds
                    y_min = max(0, y - patch_size)
                    y_max = min(img.shape[0], y + patch_size)
                    x_min = max(0, x - patch_size)
                    x_max = min(img.shape[1], x + patch_size)
                    
                    # Gennemsnitlig farve i patch
                    patch = img[y_min:y_max, x_min:x_max]
                    colors[i, j] = patch.mean(axis=(0, 1))
        
        return colors


    print("Computing patch colors...")
    colors_P = get_patch_colors(img_P, C_P_np, patch_size)
    colors_Q = get_patch_colors(img_Q, C_Q_np, patch_size)


    k = 4
    blur_colors_P = get_patch_colors(img_P, C_P_np, k * s)
    blur_colors_Q = get_patch_colors(img_Q, C_Q_np, k * s)

    hawking_colors_P = get_patch_colors(img_P, C_P_np, 0)
    hawking_colors_Q = get_patch_colors(img_Q, C_Q_np, 0)

    print("Visualizing sampled patches...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 6))
    # plt.figure(figsize=(8, 4.5))

    # Original billeder
    """plt.imshow(img_P)
    plt.title('Original Image P')
    # plt.axis('off')"""

    # Sampled patches som pixeleret billede
    axes[0, 0].imshow(img_P)
    axes[0, 0].set_title('Original Image P')
    axes[0, 0].axis('off')

    # # Sampled patches som pixeleret billede
    axes[1, 0].imshow(colors_P.astype(np.uint8), interpolation='nearest')
    axes[1, 0].set_title(f'P patch == s // 2')
    axes[1, 0].axis('off')

    # Sampled patches som pixeleret billede
    axes[0, 1].imshow(img_Q)
    axes[0, 1].set_title('Original Image Q')
    axes[0, 1].axis('off')

    # # Sampled patches som pixeleret billede
    axes[1, 1].imshow(colors_Q.astype(np.uint8), interpolation='nearest')
    axes[1, 1].set_title(f'Q patch == s // 2')
    axes[1, 1].axis('off')

    # # Sampled patches som pixeleret billede
    """axes[1, 1].imshow(blur_colors_P.astype(np.uint8), interpolation='nearest')
    axes[1, 1].set_title(f'patch == {k} * s')
    axes[1, 1].axis('off')

    # # Sampled patches som pixeleret billede
    axes[0, 1].imshow(hawking_colors_P.astype(np.uint8), interpolation='nearest')
    axes[0, 1].set_title(f'patch == 0')
    axes[0, 1].axis('off')"""

    """plt.scatter(C_P_np[:, :, 0].flatten(),
                    C_P_np[:, :, 1].flatten(),
                    c='k',
                    s=3,
                    marker='s')
                    # c=colors_P[:, :, 0])"""

    plt.subplots_adjust(wspace=3.0, hspace=3.0)
    plt.tight_layout()

    plt.close()

    plt.show()


    # ----------------------

    plt.figure(figsize=(8, 4.5))

    plt.scatter(C_P_np[:, :, 0].flatten(),
                C_P_np[:, :, 1].flatten(),
                c=colors_P.reshape(-1, 3) / 255.0,
                s=50,
                marker='o')

    plt.scatter(C_Q_np[:, :, 0].flatten(),
                C_Q_np[:, :, 1].flatten(),
                c=colors_Q.reshape(-1, 3) / 255.0,
                s=5,
                marker='o')


    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.close()
    plt.show()

    # sys.exit()


    """plt.figure(figsize=(10, 8))
    plt.scatter(C_P_np[:, :, 0].flatten(),
                C_P_np[:, :, 1].flatten(),
                c=colors_P.reshape(-1, 3)/255.0,
                s=50,
                marker='s')
                # c=colors_P[:, :, 0])

    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.show()

    sys.exit()"""

    # K_np = np.zeros((x_P.size, x_Q.size))
    
    # counter = 0
    # counter_total = 0
    # for i in range(x_P.shape[0]):
    #     for j in range(x_P.shape[1]):
    #         for v in range(x_Q.shape[0]):
    #             for w in range(x_Q.shape[1]):
    #                 row_P, col_P = C_P_np[i, j]
    #                 row_Q, col_Q = C_Q_np[v, w]
    #                 color_P, color_Q = img_P[col_P, row_P], img_Q[col_Q, row_Q]
    #                 color_diff = np.linalg.norm(color_P - color_Q)
    #                 if color_diff < 10.0:
    #                     print(color_diff)
    #                     K_np[x_P.shape[0] * j + i, x_Q.shape[0] * w + v] = 1.0
    #                     counter += 1
    #                 counter_total += 1
                    
    # print(f"counter: {counter}")
    # print(f"counter total: {counter_total}")
    # print(f"andel: {counter / counter_total}")

    
    # Flatten til (N, 3) og (M, 3)
    colors_P_flat = colors_P.reshape(-1, 3)  # (N, 3)
    colors_Q_flat = colors_Q.reshape(-1, 3)  # (M, 3)



    # METODE 2: Vectoriseret beregning af farveforskelle
    print("Computing color differences (vectorized)...")
    # Broadcasting: (N, 1, 3) - (1, M, 3) = (N, M, 3)
    color_diffs = colors_P_flat[:, None, :] - colors_Q_flat[None, :, :]
    color_distances = np.linalg.norm(color_diffs, axis=2)  # (N, M)
    
    # Sæt K baseret på farveafstand
    threshold = 1.0  # Juster denne værdi
    K_np = np.where(color_distances < threshold, 1.0, 0.0)

    # Alternativt: brug en smooth vægtning i stedet for hard threshold
    # K_np = np.exp(-color_distances / 30.0)  # Gaussian-lignende vægt
    

    print(f"K_np shape: {K_np.shape}")
    print(f"Number of matches (K > 0): {np.sum(K_np > 0)}")
    print(f"Total possible pairs: {K_np.size}")
    print(f"Match percentage: {100 * np.sum(K_np > 0) / K_np.size:.2f}%")
    
    # Hvis for få matches, juster threshold
    if np.sum(K_np > 0) < 10:
        print("WARNING: Very few matches found! Consider increasing threshold.")
    

    # Flatten koordinater
    C_P_np = C_P_np.reshape(-1, 2)
    C_Q_np = C_Q_np.reshape(-1, 2)

    # output1, output2 = optimize_PQ(C_P_np, C_Q_np, K_np, P_init=None, Q_init=None, lr=1e-0, steps=20000, device='cpu')

    # print(f"output 1: {output1}")
    # print()
    # print()
    # print()
    # print(f"output 2: {output2}")

    # Optimizer
    """print("\nStarting optimization...")
    output1, output2 = optimize_PQ(
        C_P_np, C_Q_np, K_np, 
        P_init=None, Q_init=None, 
        lr=1e-1, steps=20000, device='cpu'
    )"""


    print("hhhhhhheeeeeeeeeeeyyyyyyyyy")


    dtype = torch.float64
    device = torch.device('cpu')

    P = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)

    P = torch.nn.Parameter(P)
    # Q = torch.nn.Parameter(Q)

    C_P = torch.tensor(C_P_np, dtype=dtype, device=device)
    C_Q = torch.tensor(C_Q_np, dtype=dtype, device=device)
    K = torch.tensor(K_np, dtype=dtype, device=device)

    xyz_coords = R2toR3_torch(C_P, P)  # == M_pts

    xyz_flat = xyz_coords.reshape(-1, 3)

    colors_flat = colors_P.reshape(-1, 3)

    # viewer = FirstPersonViewer()
    # viewer.add_axes(length=0.5)

    # Tilføj billede 1
    num_points_minecraft = 1000
    # viewer.add_points(xyz_flat, colors_flat, point_size=2.0)
    xyz_rand = np.random.randn(3 * num_points_minecraft).reshape(-1, 3)
    
    xyz_rand[:, 1] = np.abs(xyz_rand[:, 1])  # stjerner


    colors_rand = np.array([  # np.stack
        np.random.uniform(low=0.5, high=1.0, size=num_points_minecraft),  # RED
        np.random.uniform(low=0.5, high=0.7, size=num_points_minecraft),  # GREEN
        np.random.uniform(low=0.0, high=1.0, size=num_points_minecraft)   # BLUE
    ]).T
    

    export_pointcloud_to_json(xyz_rand, colors_rand, "pointcloud.json")

    sys.exit()

    # viewer.add_points(xyz_rand, colors_rand, size=10.0)

    # viewer.add_line(np.array([0, -2, 0]), np.array([1, -2, 0]), color=np.array([1, 1, 1]))
    # viewer.add_line(np.array([1, -2, 0]), np.array([1, -2, 1]), color=np.array([0, 1, 0]))
    # viewer.add_line(np.array([1, -2, 1]), np.array([0, -2, 1]), color=np.array([1, 0, 0]))
    # viewer.add_line(np.array([0, -2, 1]), np.array([0, -2, 0]), color=np.array([0, 0, 1]))




    # Hvis du har billede 2:
    # xyz_flat2 = xyz_coords2.reshape(-1, 3)
    # colors_flat2 = img2.reshape(-1, 3)
    # viewer.add_points(xyz_flat2, colors_flat2, point_size=2.0)

    # Start viewer
    viewer.run()


    sys.exit()

    # print("\nStarting optimization with visualization...")
    # output1, output2 = optimize_PQ_with_visualization(
    #     C_P_np, C_Q_np, K_np, 
    #     colors_P.reshape(-1, 3),  # flatten til (N, 3)
    #     colors_Q.reshape(-1, 3),  # flatten til (M, 3)
    #     P_init=None, Q_init=None, 
    #     lr=1e-1, steps=2000, viz_every=50, device='cpu'
    # )
    
    print("\nStarting optimization without visualization...")
    output1, output2 = optimize_PQ(
        C_P_np, C_Q_np, K_np, 
        P_init=None, Q_init=None,
        lr=1e-2, steps=2000, device='cpu'
    )

    print(f"\nOptimized P: {output1}")
    print(f"Optimized Q: {output2}")


# output 1: [ -7.60332594  24.77911865 189.55101532]

# output 2: [ 26.05368988 -30.29259552 195.37582279]



    sys.exit()


    print("O")
    H_P, W_P = 3, 4  # eksempel
    y_P, x_P = np.mgrid[0:H_P, 0:W_P]   # giver to arrays med form (H, W)
    C = np.stack([x_P, y_P], axis=-1)  # shape: (H, W, 2)
    print("   K")

    C_P_np = np.array([
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[0, 1], [1, 1], [2, 1], [3, 1]],
        [[0, 2], [1, 2], [2, 2], [3, 2]]
    ])

    C_Q_np = np.array([
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[0, 1], [1, 1], [2, 1], [3, 1]],
        [[0, 2], [1, 2], [2, 2], [3, 2]]
    ])

    # Flatten grid'ene fra (3, 4, 2) til (12, 2)
    C_P_np = C_P_np.reshape(-1, 2)  # (12, 2)
    C_Q_np = C_Q_np.reshape(-1, 2)  # (12, 2)


    # C_P_np = np.array([
    #     [[0, 0], [1, 0], [2, 0]]
    # ])

    # C_Q_np = np.array([
    #     [[0, 0], [1, 0], [2, 0]]
    # ])

    K_np = np.ones((3*4,3*4))




    output1, output2 = optimize_PQ(C_P_np, C_Q_np, K_np, P_init=None, Q_init=None, lr=1e-2, steps=20000, device='cpu')

    print(f"output 1: {output1}")
    print()
    print()
    print()
    print(f"output 2: {output2}")

    sys.exit()

    # Indstillinger
    image_path_P = "my_module/frame1.jpg"  # Ret denne sti
    image_path2 = "my_module/frame2.jpg"  # Ret denne sti
    h_fov_deg = 150.0  # Horizontal field of view i grader
    h_fov_deg2 = 150.0  # Horizontal field of view i grader
    
    # Projicér billedet til kuglekoordinater
    xyz_coords, mask, img_P = pinhole_to_sphere_coordinates(image_path_P, h_fov_deg)
    xyz_coords2, mask2, img2 = pinhole_to_sphere_coordinates(image_path2, h_fov_deg=h_fov_deg2, dx=1.5)
    
    print(xyz_coords2.shape)
    print("pizza")

    print(f"Billede størrelse: {img_P.shape[0]}H x {img_P.shape[1]}W")
    print(f"Billede2 størrelse: {img2.shape[0]}H x {img2.shape[1]}W")
    print(f"Koordinat array shape: {xyz_coords.shape}")
    print(f"Koordinat array2 shape: {xyz_coords2.shape}")
    print(f"\nEksempel på koordinater:")
    print(f"Pixel (0, 0) img1: {xyz_coords[0, 0]}")
    print(f"Pixel (0, 0) img2: {xyz_coords2[0, 0]}")
    print(f"Pixel (centrum) img1: {xyz_coords[img_P.shape[0]//2, img_P.shape[1]//2]}")
    print(f"Pixel (centrum) img2: {xyz_coords2[img2.shape[0]//2, img2.shape[1]//2]}")
    
    # Verificer at alle punkter ligger på enhedskuglen
    distances = np.sqrt(np.sum(xyz_coords**2, axis=-1))
    distances2 = np.sqrt(np.sum(xyz_coords2**2, axis=-1))
    print(f"\nAfstand fra origo img1 (skal være ~1.0): min={distances.min():.6f}, max={distances.max():.6f}")
    print(f"\nAfstand fra origo img2 (skal være ~1.0): min={distances2.min():.6f}, max={distances2.max():.6f}")
    
    xf, xt = 0, 200
    yf, yt = 0, 200

    xf2, xt2 = 0,10000#200, 400
    yf2, yt2 = 0,10000#200, 400


    xyz_coords = xyz_coords[xf:xt, yf:yt, :]
    img_P = img_P[xf:xt, yf:yt, :]

    xyz_coords2 = xyz_coords2[xf2:xt2, yf2:yt2, :]
    img2 = img2[xf2:xt2, yf2:yt2, :]


    # Visualiser (bruger kun hver 10. pixel for hastighed)

    visualize_multiple_sphere_projection([xyz_coords, xyz_coords2], [img_P, img2], sample_step=3)
    # visualize_multiple_sphere_projection([xyz_coords], [img], sample_step=5)
    
    # visualize_sphere_projection(xyz_coords, img, sample_step=5)
    # visualize_sphere_projection(xyz_coords2, img2, sample_step=5)
    
    # Gem koordinaterne hvis ønsket
    # np.save("sphere_coordinates.npy", xyz_coords)
    # print("Koordinater gemt til 'sphere_coordinates.npy'")




