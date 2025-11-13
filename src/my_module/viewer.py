import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import ctypes
import torch


# --- hjælpefunktioner til shader ---
def compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        err = glGetShaderInfoLog(shader).decode()
        raise RuntimeError("Shader compile error: " + err)
    return shader

def create_program(vsrc, fsrc):
    vs = compile_shader(vsrc, GL_VERTEX_SHADER)
    fs = compile_shader(fsrc, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        err = glGetProgramInfoLog(prog).decode()
        raise RuntimeError("Program link error: " + err)
    glDeleteShader(vs); glDeleteShader(fs)
    return prog


# --- shaders ---
vertex_shader_src = """
#version 120
attribute vec3 a_pos;
attribute vec3 a_col;
attribute float a_size;
varying vec3 v_col;
void main() {
    v_col = a_col;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 1.0);
    gl_PointSize = a_size;   // størrelse i pixels
}
"""

fragment_shader_src = """
#version 120
varying vec3 v_col;
void main() {
    // gl_PointCoord er i [0,1]^2 - centret er (0.5,0.5)
    vec2 c = gl_PointCoord - vec2(0.5);
    if (length(c) > 0.5) discard;   // gør punktet cirkulært
    gl_FragColor = vec4(v_col, 1.0);
}
"""

# ---------- SHADER-KILDEKODE ----------
VERTEX_SHADER = """
#version 120
attribute vec3 a_pos;
attribute vec3 a_col;
attribute float a_size;
varying vec3 v_col;
void main() {
    v_col = a_col;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 1.0);
    gl_PointSize = a_size;
}
"""

FRAGMENT_SHADER = """
#version 120
varying vec3 v_col;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (length(c) > 0.5) discard;  // gør punkterne runde
    gl_FragColor = vec4(v_col, 1.0);
}
"""

class FirstPersonViewer:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D First Person Viewer")
        
        # Camera setup (observer at origin)
        self.yaw = 0.0    # rotation around y-axis
        self.pitch = 0.0  # rotation around x-axis
        
        # Mouse control
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.mouse_sensitivity = 0.2
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(70, (width / height), 0.01, 50.0)  # FOV=70 degrees
        glMatrixMode(GL_MODELVIEW)
        
        self.use_shaders = True
        try:
            self.point_prog = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
            # attribute locations (kan spørges dynamisk, men simple approach:)
            self.a_pos = glGetAttribLocation(self.point_prog, b"a_pos")
            self.a_col = glGetAttribLocation(self.point_prog, b"a_col")
            self.a_size = glGetAttribLocation(self.point_prog, b"a_size")
            glEnable(GL_PROGRAM_POINT_SIZE)  # gør at gl_PointSize virker i shader
        except Exception as e:
            print("Shader init failed, falling back to immediate mode:", e)
            self.use_shaders = False

        # Store objects to render
        self.points = []
        self.lines = []

    def add_points(self, xyz_coords, colors, size=10.0):
        if isinstance(xyz_coords, torch.Tensor):
            xyz = xyz_coords.detach().cpu().numpy().astype(np.float32)
        else:
            xyz = np.asarray(xyz_coords, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be (N,3)")
        if colors.shape[0] != xyz.shape[0]:
            raise ValueError("colors must have same length as xyz")

        if colors.max() > 1.0:
            colors = colors / 255.0
            colors = colors.astype(np.float32)

        # sizes: per-vertex eller enkelt
        if np.isscalar(size):
            sizes = np.full((len(xyz),), float(size), dtype=np.float32)
        else:
            sizes = np.asarray(size, dtype=np.float32)
            assert sizes.shape[0] == xyz.shape[0]

        # Opret VBO'er og upload data — send NumPy-array direkte (PyOpenGL håndterer pointer)
        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, xyz, GL_STATIC_DRAW)

        vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
        glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)

        vbo_size = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_size)
        glBufferData(GL_ARRAY_BUFFER, sizes, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.points.append({
            "vbo_pos": vbo_pos,
            "vbo_col": vbo_col,
            "vbo_size": vbo_size,
            "count": len(xyz)
        })


    # def add_points(self, xyz_coords, colors, size=10.0):
    #     """
    #     Add points to render
    #     xyz_coords: (N, 3) numpy array
    #     colors: (N, 3) numpy array with RGB values [0-255] or [0-1]
    #     """
    #     if isinstance(xyz_coords, torch.Tensor):
    #         xyz = xyz_coords.detach().cpu().numpy()
    #     else:
    #         xyz = np.asarray(xyz_coords)
    #     colors = np.asarray(colors)
    #     if xyz.ndim != 2 or xyz.shape[1] != 3:
    #         raise ValueError("xyz must be (N,3)")
    #     if colors.shape[0] != xyz.shape[0]:
    #         raise ValueError("colors must have same length as xyz")
        
    #     # Normalize colors if needed
    #     if colors.max() > 1.0:
    #         colors = colors / 255.0
        
    #     # størrelser - enten fast eller per-vertex
    #     if np.isscalar(size):
    #         sizes = np.full((len(xyz),), float(size), dtype=np.float32)
    #     else:
    #         sizes = np.asarray(size, dtype=np.float32)
    #         assert sizes.shape[0] == xyz.shape[0]

    #     # Opret VBO'er (één gang per dataset) og upload data
    #     vbo_pos = glGenBuffers(1)
    #     glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    #     glBufferData(GL_ARRAY_BUFFER, xyz.nbytes, xyz.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)

    #     vbo_col = glGenBuffers(1)
    #     glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
    #     glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)

    #     vbo_size = glGenBuffers(1)
    #     glBindBuffer(GL_ARRAY_BUFFER, vbo_size)
    #     glBufferData(GL_ARRAY_BUFFER, sizes.nbytes, sizes.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)

    #     # unbind for sikkerhed
    #     glBindBuffer(GL_ARRAY_BUFFER, 0)

    #     # Gem alt i self.points så vi kan tegne senere uden at gen-oprette buffere
    #     self.points.append({
    #         "vbo_pos": vbo_pos,
    #         "vbo_col": vbo_col,
    #         "vbo_size": vbo_size,
    #         "count": len(xyz)
    #     })
        
    #     """self.points.append({
    #         'xyz': xyz,
    #         'colors': colors,
    #         'size': size
    #     })"""
    
    def add_line(self, start, end, color=(1, 1, 1)):
        """Add a line from start to end"""
        self.lines.append({
            'start': np.array(start),
            'end': np.array(end),
            'color': np.array(color)
        })
    
    def add_axes(self, length=1.0):
        """Add RGB axes (X=red, Y=green, Z=blue)"""
        self.add_line([0, 0, 0], [length, 0, 0], color=(1, 0, 0))  # X axis
        self.add_line([0, 0, 0], [0, length, 0], color=(0, 1, 0))  # Y axis
        self.add_line([0, 0, 0], [0, 0, length], color=(0, 0, 1))  # Z axis
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Flyt scenen væk fra kameraet så punkter ligger i view-frustum
        glTranslatef(0.0, 0.0, -5.0)   # juster -5.0 efter behov

        # Apply camera rotation (observer stays at origin after translate)
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw, 0, -1, 0)

        # Use shader program
        if self.use_shaders:
            glUseProgram(self.point_prog)

            for pset in self.points:
                count = pset["count"]

                # POS attribute
                if self.a_pos >= 0:
                    glBindBuffer(GL_ARRAY_BUFFER, pset["vbo_pos"])
                    glEnableVertexAttribArray(self.a_pos)
                    glVertexAttribPointer(self.a_pos, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

                # COL attribute
                if self.a_col >= 0:
                    glBindBuffer(GL_ARRAY_BUFFER, pset["vbo_col"])
                    glEnableVertexAttribArray(self.a_col)
                    glVertexAttribPointer(self.a_col, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

                # SIZE attribute
                if self.a_size >= 0:
                    glBindBuffer(GL_ARRAY_BUFFER, pset["vbo_size"])
                    glEnableVertexAttribArray(self.a_size)
                    glVertexAttribPointer(self.a_size, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                else:
                    # fallback: hvis shader ikke har a_size, sæt global point size
                    glPointSize(5.0)

                # Tegn
                glDrawArrays(GL_POINTS, 0, count)

                # Disable attribs for dette sæt
                if self.a_pos >= 0:
                    glDisableVertexAttribArray(self.a_pos)
                if self.a_col >= 0:
                    glDisableVertexAttribArray(self.a_col)
                if self.a_size >= 0:
                    glDisableVertexAttribArray(self.a_size)

            # unbind og slut program
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glUseProgram(0)
        else:
            # fallback immediate mode (sanity-check)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, -2.0)
            glEnd()

        # Draw lines (unchanged)
        glBegin(GL_LINES)
        for line in self.lines:
            glColor3fv(line['color'])
            glVertex3fv(line['start'])
            glVertex3fv(line['end'])
        glEnd()

        pygame.display.flip()


    # def render(self):
    #     """Render all objects"""
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     glLoadIdentity()
        
    #     # Apply camera rotation (observer stays at origin)
    #     glRotatef(-self.pitch, 1, 0, 0)  # pitch (look up/down)
    #     glRotatef(-self.yaw, 0, -1, 0)    # yaw (look left/right)

    #     # Brug shaderprogrammet til punkter
    #     glUseProgram(self.point_prog)
        
    #     # Draw points
    #     """for point_set in self.points:
    #         glPointSize(point_set['size'])
    #         glBegin(GL_POINTS)
    #         for i in range(len(point_set['xyz'])):
    #             glColor3fv(point_set['colors'][i])
    #             glVertex3fv(point_set['xyz'][i])
    #         glEnd()"""
        
    #     # Draw points (shader path)
    #     for pset in self.points:
    #         count = pset["count"]

    #         # xyz = np.asarray(pset['xyz']).astype(np.float32)
    #         # cols = np.asarray(pset['colors']).astype(np.float32)
    #         # sizes = np.full((len(xyz),), float(pset['size']), dtype=np.float32)

    #         if self.use_shaders:
    #             # Interleave eller send som separate VBO'er (her: separate)
    #             # pos VBO
    #             # POS
    #             if self.a_pos >= 0:
    #                 # vbo_pos = glGenBuffers(1)
    #                 glBindBuffer(GL_ARRAY_BUFFER, pset["vbo_pos"])
    #                 glVertexAttribPointer(self.a_pos, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    #                 # glEnableVertexAttribArray(self.a_pos)
    #                 # glBufferData(GL_ARRAY_BUFFER, xyz, GL_STATIC_DRAW)

    #             # COL
    #             if self.a_col >= 0:
    #                 glBindBuffer(GL_ARRAY_BUFFER, pset["vbo_col"])
    #                 glEnableVertexAttribArray(self.a_col)
    #                 glVertexAttribPointer(self.a_col, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    #                 # vbo_col = glGenBuffers(1)
    #                 # glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
    #                 # glBufferData(GL_ARRAY_BUFFER, cols, GL_STATIC_DRAW)

    #             # SIZE
    #             if self.a_size >= 0:
    #                 glBindBuffer(GL_ARRAY_BUFFER, pset["vbo_size"])
    #                 glEnableVertexAttribArray(self.a_size)
    #                 glVertexAttribPointer(self.a_size, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    #             else:
    #                 # Hvis shader ikke har a_size, fallback: sæt glPointSize( single_size )
    #                 pass

    #             """vbo_size = glGenBuffers(1)
    #             glBindBuffer(GL_ARRAY_BUFFER, vbo_size)
    #             glBufferData(GL_ARRAY_BUFFER, sizes, GL_STATIC_DRAW)"""

    #             # glUseProgram(self.point_prog)
    #             # # enable and set pointers
    #             # glEnableVertexAttribArray(self.a_pos)
    #             # glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    #             # glVertexAttribPointer(self.a_pos, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    #             # glEnableVertexAttribArray(self.a_col)
    #             # glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
    #             # glVertexAttribPointer(self.a_col, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    #             # glEnableVertexAttribArray(self.a_size)
    #             # glBindBuffer(GL_ARRAY_BUFFER, vbo_size)
    #             # glVertexAttribPointer(self.a_size, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    #             glDrawArrays(GL_POINTS, 0, count)

    #             # cleanup
    #             # glDisableVertexAttribArray(self.a_pos)
    #             # glDisableVertexAttribArray(self.a_col)
    #             # glDisableVertexAttribArray(self.a_size)
    #             # glUseProgram(0)

    #             # cleanup for dette sæt
    #             if self.a_pos >= 0:
    #                 glDisableVertexAttribArray(self.a_pos)
    #             if self.a_col >= 0:
    #                 glDisableVertexAttribArray(self.a_col)
    #             if self.a_size >= 0:
    #                 glDisableVertexAttribArray(self.a_size)

    #             glBindBuffer(GL_ARRAY_BUFFER, 0)
    #             glUseProgram(0)
    #             # glDeleteBuffers(3, [vbo_pos, vbo_col, vbo_size])
    #         else:
    #             # fallback immediate mode
    #             glPointSize(pset['size'])
    #             glBegin(GL_POINTS)
    #             for i in range(len(xyz)):
    #                 glColor3fv(colors[i])
    #                 glVertex3fv(xyz[i])
    #             glEnd()
        
    #     # Draw lines
    #     glBegin(GL_LINES)
    #     for line in self.lines:
    #         glColor3fv(line['color'])
    #         glVertex3fv(line['start'])
    #         glVertex3fv(line['end'])
    #     glEnd()

    #     pygame.display.flip()
    
    def handle_mouse(self):
        """Handle mouse movement for camera rotation"""
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        
        self.yaw += mouse_dx * self.mouse_sensitivity
        self.pitch -= mouse_dy * self.mouse_sensitivity
        
        # Clamp pitch to avoid flipping
        self.pitch = max(-89, min(89, self.pitch))
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("Controls:")
        print("  Mouse: Look around")
        print("  ESC: Exit")
        
        while running:
            print("attr locs: a_pos", self.a_pos, "a_col", self.a_col, "a_size", self.a_size)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            self.handle_mouse()
            self.render()
            clock.tick(60)  # 60 FPS
        
        
        pygame.quit()


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    viewer = FirstPersonViewer()
    
    # Add coordinate axes
    viewer.add_axes(length=1.0)
    
    # Example: Add points on a sphere
    n_points = 1000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 2.0  # radius
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    xyz = np.stack([x, y, z], axis=1)
    colors = np.random.rand(n_points, 3)
    
    viewer.add_points(xyz, colors, size=3.0)
    
    # Run the viewer
    viewer.run()


# ========== INTEGRATION WITH YOUR CODE ==========
"""
# I din cli.py, efter du har xyz_coords og img fra pinhole_to_sphere_coordinates:

from my_module.viewer import FirstPersonViewer

# Flatten coordinates og farver
xyz_flat = xyz_coords.reshape(-1, 3)
colors_flat = img.reshape(-1, 3)

# Lav viewer
viewer = FirstPersonViewer()
viewer.add_axes(length=0.5)

# Tilføj billede 1
viewer.add_points(xyz_flat, colors_flat, point_size=2.0)

# Hvis du har billede 2:
# xyz_flat2 = xyz_coords2.reshape(-1, 3)
# colors_flat2 = img2.reshape(-1, 3)
# viewer.add_points(xyz_flat2, colors_flat2, point_size=2.0)

# Start viewer
viewer.run()
"""
