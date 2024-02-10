export MS2_ASSET_DIR=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data

gpu_id=0
policy_model=octo-base
ckpt_path=/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/
scene_name=bridge_table_1_v1
rgb_overlay_path=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/real_impainting/bridge_real_eval_1.png

# CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
#   --robot widowx --policy-setup widowx_bridge \
#   --control-freq 5 --sim-freq 500 --max-episode-steps 50 \
#   --env-name PutSpoonOnTableClothBakedTexInScene-v0 --scene-name ${scene_name} \
#   --rgb-overlay-path ${rgb_overlay_path} \
#   --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 50 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 50 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;


policy_model=octo-small

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 50 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;