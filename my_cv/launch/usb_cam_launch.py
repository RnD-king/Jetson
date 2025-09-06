from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    gst = (
      'v4l2src device=/dev/video-webcam do-timestamp=true ! '
      'video/x-h264,stream-format=byte-stream,alignment=au,framerate=15/1,width=640,height=480 ! '
      'h264parse ! nvv4l2decoder ! '
      'nvvidconv ! video/x-raw,format=NV12 ! '
      'videoconvert ! video/x-raw,format=RGB'
    )
    return LaunchDescription([
        Node(
            package='gscam',
            executable='gscam_node',
            namespace='webcam',
            name='gscam_webcam',
            output='screen',
            parameters=[{
                'gscam_config': gst,
                'frame_id': 'webcam_optical_frame',
                'image_encoding': 'rgb8',
                'sync_sink': False,
            }]
        )
    ])
 # 여기 안 쓴다
