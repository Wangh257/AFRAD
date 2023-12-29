obj_id=1
seperate=1
#train_name='201'
train_name='all'

#train_dir='/home/wangh20/path/mvt-ad/'
train_dir='/home/wangh20/path/HANIT/Dongci/'
#train_dir='/home/wangh20/path/ciwa_ok'
#train_dir='/home/wangh20/path/ciwa_public'

anomaly_list='/home/wangh20/path/dtd/source_img.lst'
#anomaly_list='/home/wangh20/path/ciwa_ok/140+280/ok.lst'
gpu_id=1
lr=0.0001
bs=4
epochs=1000
#checkpoint_path='./chechpoints/CBAM_attention/mvt/sep_1/'
checkpoint_path='./chechpoints/HANIT/Dongci/0100/'
log_path='./log_path/HANIT/Dongci/0100/'

        python ../train_attention.py \
        --obj_id=$obj_id \
        --gpu_id=$gpu_id \
        --lr=$lr \
        --train_name=$train_name \
        --train_dir=$train_dir \
        --anomaly_list=$anomaly_list \
        --bs=$bs \
        --epochs=$epochs \
        --anomaly_source_path=$anomaly_source_path \
        --checkpoint_path=$checkpoint_path \
        --log_path=$log_path \
        --seperate=$seperate

