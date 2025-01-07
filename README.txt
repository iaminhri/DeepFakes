Hello,

I have included the image files, .ipynb (google colab) files, and project report on the root directory.

The datasets are not included in here. However, the dataset can be found in the data/ folder. 

If you want to run the project, there is a Python Files folder.
The architecture is defined as 3x128x128 in this 128x128_deepfakesgans.py file.

to run the python files, please run the following commands.

# note that python version needs to be less than 3.13
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt.

If you want to run the trained models and see output:
1. download the dataset.
2. savedModel -> this folder contains the trained parameters of the generator and discriminator model and the optimizers as well.
2. modify the generator_path, discriminator_path, g_optimizer_path, d_optimizer_path within the runTrainedModel.py
3. run

In the IPYNB files folder, there is python file called, 128x128_Trained_DCGAN.py this file,
evaluates the discriminator's accuracy for real and fake images

Thank you for your time.

Best Regards




