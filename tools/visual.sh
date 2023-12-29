obj_id=0
test_name='MT_Free'

#test_name='test'

test_name='carpet'
test_name='leather'

test_name='bottle'
test_name='capsule'
test_name='101'
#test_dir='/home/wangh20/path/mvt-ad'
test_dir='/home/wangh20/path/HANIT/Dongci'
#test_dir='/home/wangh20/path/ciwa_ok'
#test_dir='/home/wangh20/path/ciwa_public'
base_name='0.0001_700_bs8_700'
#checkpoint_dir='/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/with_gan/mvt/no_dis'
#checkpoint_dir='/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/with_gan/mvt/seperate_1'
#checkpoint_dir='/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/gan_attention/mvt/no_dis'
#checkpoint_dir='/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/attention/mvt/no_dis/'
checkpoint_dir='/home/wangh20/projects/DRAEM_attention/tools/chechpoints/Spa_attention/mvt/noly_encoder'
checkpoint_dir='/home/wangh20/projects/DRAEM_attention/tools/chechpoints/Spa_attention/mvt/no_dis'
checkpoint_dir='/home/wangh20/projects/DRAEM_attention/tmp/capsule/Spa_attention/no_dis/0010'
checkpoint_dir='/home/wangh20/projects/DRAEM_attention/tools/chechpoints/HANIT/Dongci/0100'
gpu_id=0
#save_dir='./img/ciwa_public/seperate_1/differ_epoch/800'
#save_dir='./img_tmp/Spa_attention/no_dis/0010'
save_dir='./result/HANIT/Dongci/0100'

      python ../visual.py \
      --obj_id=$obj_id \
      --gpu_id=$gpu_id \
      --test_dir=$test_dir \
      --checkpoint_dir=$checkpoint_dir \
      --base_name=$base_name \
      --test_name=$test_name \
      --save_dir=$save_dir

