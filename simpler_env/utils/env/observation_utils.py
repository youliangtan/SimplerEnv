def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        # Access robot_uid through unwrapped env if wrapped
        robot_uid = getattr(env, 'robot_uid', None)
        if robot_uid is None and hasattr(env, 'unwrapped'):
            robot_uid = getattr(env.unwrapped, 'robot_uid', None)
        
        if robot_uid and "google_robot" in robot_uid:
            camera_name = "overhead_camera"
        elif robot_uid and "widowx" in robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]
