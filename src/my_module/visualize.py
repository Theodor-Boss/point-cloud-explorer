import matplotlib.pyplot as plt
import numpy as np

def visualize_multiple_sphere_projection(lst_xyz_coords, lst_img, sample_step=10):
    """
    Visualiserer projektionen på enhedskuglen med farver fra billedet.
    
    Args:
        xyz_coords: (H, W, 3) array med 3D koordinater
        img: (H, W, 3) array med RGB billede
        sample_step: Sample kun hver N'te pixel for at reducere antallet af punkter
    """
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    for (xyz_coords, img) in zip(lst_xyz_coords, lst_img):
        H, W = xyz_coords.shape[:2]
        
        # Sample pixels
        xyz_sampled = xyz_coords[::sample_step, ::sample_step].reshape(-1, 3)
        colors_sampled = img[::sample_step, ::sample_step].reshape(-1, 3) / 255.0
        
        # Original billede
        # ax1.imshow(img)
        
        # 3D projektion
        ax2.scatter(xyz_sampled[:, 0], 
                    xyz_sampled[:, 1],
                    xyz_sampled[:, 2], 
                    c=colors_sampled, 
                    s=4, 
                    alpha=0.6)
        
        # print(xyz_sampled[:, 0].shape)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Projektion på enhedskuglen')

    ax1.set_title("Original billede")
    ax1.axis('off')

    # Sæt lige store akser
    ax2.set_box_aspect([1,1,1])
    max_range = 1.0
    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])

    plt.tight_layout()
    plt.show()


def simple_visualize_plane_projection(xyz_coords):
    xyz = np.asarray(xyz_coords)
    
    if xyz.ndim == 3 and xyz.shape[2] == 3:
        pts = xyz.reshape(-1, 3)
    elif xyz.ndim == 2 and xyz.shape[1] == 3:
        pts = xyz
    else:
        raise ValueError("expected shape (ny, nx, 3) or (N, 3)")
    
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c="C0", s=6, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Projektion af plan')
    
    # set box aspect and auto-scale to data
    ax.set_box_aspect([1,1,1])
    # use actual data range (with a little padding)
    pad = 0.05 * max(xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min(), 1.0)
    ax.set_xlim([xs.min()-pad, xs.max()+pad])
    ax.set_ylim([ys.min()-pad, ys.max()+pad])
    ax.set_zlim([zs.min()-pad, zs.max()+pad])

    plt.tight_layout()
    plt.show()



def visualize_plane_projection(lst_xyz_coords, lst_img, sample_step=10):
    """
    Visualiserer projektionen på enhedskuglen med farver fra billedet.
    
    Args:
        xyz_coords: (H, W, 3) array med 3D koordinater
        img: (H, W, 3) array med RGB billede
        sample_step: Sample kun hver N'te pixel for at reducere antallet af punkter
    """
    fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    for (xyz_coords, img) in zip(lst_xyz_coords, lst_img):
        H, W = xyz_coords.shape[:2]
        
        # Sample pixels
        xyz_sampled = xyz_coords[::sample_step, ::sample_step].reshape(-1, 3)
        colors_sampled = img[::sample_step, ::sample_step].reshape(-1, 3) / 255.0
        
        # Original billede
        # ax1.imshow(img)
        
        # 3D projektion
        ax2.scatter(xyz_sampled[:, 0], 
                    xyz_sampled[:, 1],
                    xyz_sampled[:, 2], 
                    c=colors_sampled, 
                    s=4, 
                    alpha=0.6)
        
        print(xyz_sampled[:, 0].shape)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Projektion på enhedskuglen')

    ax1.set_title("Original billede")
    ax1.axis('off')

    # Sæt lige store akser
    ax2.set_box_aspect([1,1,1])
    max_range = 1.0
    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])

    plt.tight_layout()
    plt.show()
