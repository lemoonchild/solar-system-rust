# 🚀 Solar System Simulation with 3D Spaceship 

This project is a 3D simulation of a solar system developed in **Rust**, featuring a custom spaceship that follows the camera, a starry skybox, and orbiting planets. It also includes a 3D camera with intuitive controls to explore the system.

## 🛠️ Features
- **Custom Spaceship:** A modeled and rendered spaceship with a shader blending purple and green colors, inspired by EVA-01 from Evangelion.
- **Starry Skybox:** A star-filled horizon to set the atmosphere.
- **Collision Prevention:** Prevents the spaceship and camera from passing through planets.
- **3D Camera Movement:** Free camera movement in 3D space.
- **Planetary Orbits:** Smooth simulation of planetary orbits.
- **Mouse and Scroll Controls:** Dynamic camera handling and zoom.
- **Bird Eye View:** Switch to an aerial view to visualize the entire solar system.

## 🎮 Controls

### Keyboard
- **`W` and `S`:** Rotate the camera up and down.
- **`A` and `D`:** Move the camera left and right.
- **`Q` and `E`:** Move the camera up and down (vertical axis).
- **`↑` and `↓`:** Zoom in and out.
- **`B`:** Toggle between Bird Eye View and normal view.

### Mouse
- **Drag:** Rotate the camera by dragging the mouse.
- **Scroll:** Zoom in and out.

## 📜 System Requirements
- **Rust Compiler** (Latest version recommended).
- Required Libraries:
  - `nalgebra_glm` for mathematical calculations.
  - `minifb` for window rendering.
  - `fastnoise_lite` for noise generation.


## 🎥 Demonstration Video
Link to the video: https://youtu.be/j_mEIh4oLpM 

## 📝 Additional Notes
- This project includes implementations of **frustum culling** and **backface culling** for rendering optimization.
- The spaceship shader uses linear interpolation (lerp) of purple and green colors combined with noise for a unique effect.

