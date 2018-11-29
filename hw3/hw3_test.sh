# Download models, there are a total of 6 models to download
# model Mobilenet
wget -O Mobilenet_models_e1500.pt 'https://www.dropbox.com/s/vmn3f1a9hw3mkqc/Mobilenet_models_e1500.pt?dl=1'
# model VGG
wget -O VGG_models_e2000.pt 'https://www.dropbox.com/s/5f2mom4wxt2u6al/VGG_models_e2000.pt?dl=1'
# model Kai
wget -O Kai_models_e2500.pt 'https://www.dropbox.com/s/9r4d0gnx5vufw4q/Kai_models_e2500.pt?dl=1'
# model Kai2
wget -O Kai2_models_e500.pt 'https://www.dropbox.com/s/epac64re3d5aegf/Kai2_models_e500.pt?dl=1'
# model Kai3
wget -O Kai3_models_e3000.pt 'https://www.dropbox.com/s/9qaa2l43hzi9x2o/Kai3_models_e3000.pt?dl=1'
# model Kai4
wget -O Kai4_models_e500.pt 'https://www.dropbox.com/s/86e7hb3n86f1lbf/Kai4_models_e500.pt?dl=1'

# inference testing data
# It takes about 2G and 30 seconds to run on my machine (K80)
# If CUDA out of memory, TAs can change batch size using argument '--batch_size'. If there is no GPU, type '--use_cuda' to inference with only CPU. 
python3 test.py --ensemble --models Mobilenet VGG Kai Kai2 Kai3 Kai4 --model_filenames Mobilenet_models_e1500.pt VGG_models_e2000.pt Kai_models_e2500.pt Kai2_models_e500.pt Kai3_models_e3000.pt Kai4_models_e500.pt --test_filename $1 --output $2
