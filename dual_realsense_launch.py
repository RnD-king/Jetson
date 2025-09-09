# dual_realsense_launch.py
import os
import yaml
from launch import LaunchDescription
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration

# ==== 원본 rs_launch.py와 동일한 헬퍼 ====
def yaml_to_dict(path_to_yaml):
    with open(path_to_yaml, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(p['name'], default_value=p['default'], description=p['description']) for p in parameters]
    
def set_configurable_parameters_unsuffixed(parameters, suffix):
    """
    런치 인자는 suffix 붙은 이름(예: camera_name1)으로 받고,
    노드에 넘길 params dict는 suffix를 뗀 원래 키(예: camera_name)로 만든다.
    """
    m = {}
    for p in parameters:
        suffixed = p['name']                  # 예: 'camera_name1'
        if not suffixed.endswith(suffix):
            # 방어적 처리: 혹시나 suffix가 없다면 그냥 원래 이름으로 전달
            base = suffixed
        else:
            base = suffixed[:-len(suffix)]    # 'camera_name'
        m[base] = LaunchConfiguration(suffixed)  # 값은 'camera_name1' 런치인자 바인딩
    return m

def set_configurable_parameters(parameters):
    return dict([(p['name'], LaunchConfiguration(p['name'])) for p in parameters])

def launch_setup(context, params, param_name_suffix=''):
    _config_file = LaunchConfiguration('config_file' + param_name_suffix).perform(context)
    params_from_file = {} if _config_file == "''" else yaml_to_dict(_config_file)

    lifecycle_param_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'global_settings.yaml')
    lifecycle_params = yaml_to_dict(lifecycle_param_file)
    use_lifecycle_node = lifecycle_params.get("use_lifecycle_node", False)

    _output = LaunchConfiguration('output' + param_name_suffix)
    node_action = launch_ros.actions.LifecycleNode if use_lifecycle_node else launch_ros.actions.Node
    log_message = "Launching as LifecycleNode" if use_lifecycle_node else "Launching as Normal ROS Node"

    if (os.getenv('ROS_DISTRO') == 'foxy'):
        _output = context.perform_substitution(_output)

    return [
        LogInfo(msg=f"🚀 {log_message} (cam{param_name_suffix})"),
        node_action(
            package='realsense2_camera',
            namespace=LaunchConfiguration('camera_namespace' + param_name_suffix),
            name=LaunchConfiguration('camera_name' + param_name_suffix),
            executable='realsense2_camera_node',
            parameters=[params, params_from_file],
            output=_output,
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level' + param_name_suffix)],
            emulate_tty=True,
        )
    ]

# ==== 여기부터: 두 대용 정의 ====

# 원본 configurable_parameters를 suffix 버전으로 복제/오버라이드하는 유틸
def suffixed_params(base, suffix, overrides=None):
    overrides = overrides or {}
    cloned = []
    for p in base:
        name = p['name'] + suffix
        default = overrides.get(p['name'], p['default'])
        cloned.append({'name': name, 'default': default, 'description': p['description']})
    return cloned

# 원본과 같은 파라미터 세트(필요 항목만 발췌/정렬)
BASE_PARAMS = [
    {'name':'camera_name', 'default':'camera', 'description':'camera unique name'},
    {'name':'camera_namespace','default':'', 'description':'namespace for camera'},
    {'name':'serial_no', 'default':"''", 'description':'choose device by serial number'},
    {'name':'config_file', 'default':"''", 'description':'yaml config file'},
    {'name':'json_file_path', 'default':"''", 'description':'allows advanced configuration'},
    {'name':'log_level','default':'info','description':'debug log level'},
    {'name':'output','default':'screen','description':'node output'},

    # 스트림 on/off
    {'name':'enable_color','default':'true','description':'enable color stream'},
    {'name':'enable_depth','default':'true','description':'enable depth stream'},
    {'name':'enable_infra','default':'false','description':'enable infra0 stream'},
    {'name':'enable_infra1','default':'false','description':'enable infra1 stream'},
    {'name':'enable_infra2','default':'false','description':'enable infra2 stream'},
    {'name':'enable_gyro','default':'false','description':''},
    {'name':'enable_accel','default':'false','description':''},
    {'name':'enable_motion','default':'false','description':''},

    # 프로필/포맷
    {'name':'rgb_camera.color_profile','default':'640,480,15','description':'color stream profile'},
    {'name':'rgb_camera.color_format','default':'RGB8','description':'color stream format'},
    {'name':'rgb_camera.enable_auto_exposure','default':'false','description':'color AE'},

    {'name':'depth_module.depth_profile','default':'640,480,15','description':'depth stream profile'},
    {'name':'depth_module.depth_format','default':'Z16','description':'depth stream format'},
    {'name':'depth_module.enable_auto_exposure','default':'true','description':'depth AE'},

    # 필터/기타
    {'name':'align_depth.enable','default':'false','description':'align depth to color'},
    {'name':'publish_tf','default':'true','description':''},
    {'name':'tf_publish_rate','default':'0.0','description':''},
    {'name':'pointcloud.enable','default':'false','description':''},
    {'name':'pointcloud.ordered_pc','default':'false','description':''},
]

def generate_launch_description():
    # cam1: 컬러+깊이 (정렬 on)
    cam1_overrides = {
        'camera_name': 'cam1',
        'camera_namespace': '',
        'serial_no': '"148522071908"',      # 시리얼 넘버    
        'enable_color': 'true',
        'enable_depth': 'true',
        'align_depth.enable': 'true',
        'rgb_camera.power_line_frequency': '2',
        'rgb_camera.color_profile': '640,480,15',
        'depth_module.depth_profile': '640,480,15',
        # 필요시 infra/imu 추가로 off
        #'enable_infra': 'false', 'enable_infra1': 'false', 'enable_infra2': 'false',
        #'enable_gyro': 'false', 'enable_accel': 'false', 'enable_motion':'false',
    }
    params_cam1 = suffixed_params(BASE_PARAMS, '1', cam1_overrides)

    # cam2: 컬러만 (깊이/IR/IMU off)
    cam2_overrides = {
        'camera_name': 'cam2',
        'camera_namespace': '',
        'serial_no': '"332522071721"',     # 시리얼 넘버
        'enable_color': 'true',
        'enable_depth': 'false',
        'align_depth.enable': 'false',
        'rgb_camera.color_profile': '640,480,15',
        # infra/imu 모두 off
        'enable_infra': 'false', 'enable_infra1': 'false', 'enable_infra2': 'false',
        'enable_gyro': 'false', 'enable_accel': 'false', 'enable_motion':'false',
    }
    params_cam2 = suffixed_params(BASE_PARAMS, '2', cam2_overrides)

    # LaunchArguments 선언 (cam1+cam2)
    decl = []
    decl += declare_configurable_parameters(params_cam1)
    decl += declare_configurable_parameters(params_cam2)

    # 각 카메라 노드 생성 (OpaqueFunction으로 원본 launch_setup 재사용)
    actions = []
    actions += [OpaqueFunction(function=launch_setup,
                           kwargs={'params': set_configurable_parameters_unsuffixed(params_cam1, '1'),
                                   'param_name_suffix': '1'})]
    actions += [OpaqueFunction(function=launch_setup,
                           kwargs={'params': set_configurable_parameters_unsuffixed(params_cam2, '2'),
                                   'param_name_suffix': '2'})]

    return LaunchDescription(decl + actions)

