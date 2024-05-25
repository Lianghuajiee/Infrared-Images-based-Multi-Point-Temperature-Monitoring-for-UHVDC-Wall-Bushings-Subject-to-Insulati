### Project Reproduction Instructions

This code can directly reproduce FIGURE 4, FIGURE 6, Kernel accuracy in section 3.2, and FIGURE 12.

### Runtime Environment
- Python version: 3.10.13
- Library versions: See `requirements.txt`

### File Descriptions
- `Display_images_of_the_eight_bushings`: Information on visible light and infrared images of the eight bushings.
- `Data`:
  - `Raw_data`: Essential files for running the code. Raw infrared image data of the bushings.
  - `Generated_data`: Non-essential files for running the code. Data generated during the code execution.
  - `mlp_cpu_model.pth`: Essential file for running the code. Model for recognizing the maximum and minimum temperature values in infrared images.
  - `mlp_model.pth`: Essential file for running the code. Model for recognizing the maximum and minimum temperature values in infrared images.
- `subfunction`: Essential files for running the code. Subfunctions.
- `result`: Non-essential files for running the code. Location for storing all results.
- `main.py`: Essential file for running the code. Main function.

### Code Execution Instructions
```
To reproduce all results:
  python main.py all

To reproduce the confusion matrix in FIGURE 4:
  python main.py FIGURE_4

To reproduce the accuracy results in FIGURE 6:
  python main.py FIGURE_6

To reproduce the Kernel accuracy in section 3.2:
  python main.py kernel

To reproduce the unsupervised algorithm combination results in FIGURE 12:
  python main.py FIGURE_12
```
