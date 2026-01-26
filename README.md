# Video-Special-Effects

A real-time video processing application built with C++ and OpenCV. This project captures live video input from a webcam and applies a wide variety of visual effects and filters interactively using keyboard shortcuts.

## Features

* **Real-time Processing:** Filters are applied instantly to the live video stream.
* **Interactive Controls:** Switch between filters on the fly using keyboard shortcuts.
* **Diverse Effect Library:** Includes basic color manipulations, edge detection, artistic filters, and time-displacement effects.
* **Face Detection:** Integrated Haar Cascade face detection for specific augmented reality effects.

## Dependencies

* **C++ Compiler**
* **OpenCV**
* **CMake** (for building the project)

## Installation & Build

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hyukjin17/Video-Special-Effects.git](https://github.com/hyukjin17/Video-Special-Effects.git)
    cd Video-Special-Effects
    ```

2.  **Create a build directory:** (use release config for faster real time video processing)
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    ```

3.  **Run CMake and Compile:**
    ```bash
    cmake --build build --config Release
    make
    ```

4.  **Run the Application:**
    Ensure the `haarcascade_frontalface_alt2.xml` file is in the same directory as the executable, or that the path in the code points to it correctly.
    ```bash
    ./build/project1
    ```

## Usage / Controls

Click on the "Live Video" window to ensure it is selected, then press the following keys to toggle effects:

### Basic Filters
| Key | Effect | Description |
| :--- | :--- | :--- |
| `c` | **Clear / Reset** | Returns to the original raw color video. |
| `g` | **Grayscale** | Converts video to standard grayscale. |
| `h` | **Alt Grayscale** | An alternative, custom grayscale implementation. |
| `s` | **Sepia** | Applies a vintage, warm sepia tone. |
| `b` | **Blur** | Applies a 5x5 Gaussian blur to smooth the image. |
| `r` | **Red Channel** | Extracts and displays only red colors using HSV thresholds. |
| `w` | **Mirror** | Mirrors the image with respect to the central y axis. |

### Edge Detection & Analysis
| Key | Effect | Description |
| :--- | :--- | :--- |
| `x` | **Sobel X** | Highlights vertical edges. |
| `y` | **Sobel Y** | Highlights horizontal edges. |
| `m` | **Magnitude** | Combines X and Y to show overall edge magnitude. |
| `i` | **Inv. Magnitude** | Inverted edge detection (looks like a pencil sketch). |
| `z` | **Laplacian** | Second-order derivative edge detection. |
| `e` | **Embossing** | Creates a 3D "relief" effect using a directional filter. |
| `j` | **Embossing V2** | Alternative embossing using steerable filters. |

### Artistic & Motion Effects
| Key | Effect | Description |
| :--- | :--- | :--- |
| `l` | **Blur & Quantize** | Creates a "cartoon" or posterized effect. |
| `f` | **Face Detection** | Draws a box around detected faces. |
| `t` | **Focus Mode** | Grays the background but keeps faces in color. |
| `1` | **Motion Detect** | Highlights only the pixels that are currently moving. |
| `2` | **Slit Scan** | A time-displacement "horizontal scanner" effect. |
| `3` | **Motion Blur** | Creates a trailing blur effect based on movement. |
| `4` | **Ghosting** | Simple multi-frame ghosting effect. |
| `5` | **Smooth Ghost** | Advanced ghosting with fluid decay and circular buffering. |

### UI Controls
* **Trackbars:** Some modes (like Motion Blur and Ghost) may trigger a slider at the top of the window to adjust intensity.
* **`q`**: Quit the application.

## Project Structure

* `vidDisplay.cpp`: The main entry point. Handles the video loop, keyboard input, and UI management.
* `filter.cpp`: Contains the implementation logic for all image processing functions.
* `filter.hpp`: Header file declaring the filter functions.
* `haarcascade_frontalface_alt2.xml`: Pre-trained model required for face detection features.