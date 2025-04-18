# SRCNN

This GitHub repository is an implementation of the Super-Resolution Convolutional Neural Network originally introduced by Dong et al. (https://arxiv.org/abs/1501.00092).

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- tqdm 4.30.0

The code for files stability_study.py, models.py and utils.py was adapted from files SRCNN/Torch/train.py, SRCNN/Torch/models.py and SRCNN/Torch/util.py in the publicly available GitHub repository https://github.com/Fivefold/SRCNN (Copyright (c) 2021 Jakob Essbüchl, Philipp Lehninger, Benedikt Morgenbesser), accessed on 12.06.2023.

For these files, the following copyright notice is included:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The repository additionally includes the training of the SRCNN model embedded in a stability scheme (stability_study.py) and an analysis of local stability of the model results in example images (local_stability.py). The Python-Notebooks srcnn_results_analysis.ipynb, srcnn_results_analysis_10.ipynb, and srcnn_results_analysis_10_0015.ipynb contain the analysis and creation of figures for the results obtained from the performed stability scheme. 