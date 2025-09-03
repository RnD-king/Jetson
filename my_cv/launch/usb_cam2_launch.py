from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    gst = (
      'v4l2src device=/dev/video-webcamB io-mode=2 do-timestamp=true ! '
      'queue max-size-buffers=1 leaky=downstream ! ' 
      'image/jpeg,width=640,height=480,framerate=15/1 ! '
      'jpegparse ! '
      'nvv4l2decoder mjpeg=1 enable-max-performance=1 ! '
      'queue max-size-buffers=1 leaky=downstream ! ' 
      'nvvidconv ! video/x-raw,format=RGBA,width=640,height=480 ! '
      'videoconvert ! video/x-raw,format=RGB,width=640,height=480' #CPU
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
          'reopen_on_eof': True,
        }]
      )
    ])

