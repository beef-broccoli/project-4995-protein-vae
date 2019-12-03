# protein-vae
final project for 4995 deep learning class
This project builds upon existing work: https://github.com/psipred/protein-vae

Data used for training and trained models are too big for github. Available from author upon request



Descriptions for files:

mprotein_seq.txt ==> Raw protein sequences 

conv_cvae_metal_gen.py ==> Convolutional VAE training script

metal_gen.py ==> Fully-connected VAE training script

seq_to_seq.py ==> reconstruct sequence, no metal-binding code, see script for usage, onfigured for Conv VAE

seq_to_seq.py ==> reconstruct sequence with metal-binding code, see script for usage, configured for Conv VAE

utils.py ==> helper functions

Conv CVAE metal_gen.ipynb ==> jupyter notebook, for running and testing on colab (this script is not extensively tested, might differ from actual training script/ contain errors)
