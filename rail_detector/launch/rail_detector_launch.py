from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    use_row_counter_arg = DeclareLaunchArgument(
        'use_row_counter',
        default_value='true',
        description='Start the row-counter node')

    use_rqt_arg = DeclareLaunchArgument(
        'use_rqt',
        default_value='true',
        description='Start rqt_image_view for live preview')
    
    use_gpu_arg = DeclareLaunchArgument(
    'use_gpu',
    default_value='true',
    description='Run inference on the GPU (CUDA) if true, CPU if false')

    use_row_counter = LaunchConfiguration('use_row_counter')
    use_rqt         = LaunchConfiguration('use_rqt')
    use_gpu = LaunchConfiguration('use_gpu')

    model_path = PathJoinSubstitution([
        FindPackageShare('rail_detector'),
        'models',
        'rail_detector.onnx'
    ])

    detector_node = Node(
        package='rail_detector',
        executable='rail_detector_node',
        name='rail_detector_node',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'use_gpu':    use_gpu
        }]
    )

    row_counter_node = Node(
        package='rail_detector',
        executable='row_counter_node_exec',
        name='row_counter_node',
        output='screen',
        parameters=[{
            'row_spacing':     0.60,
            'min_area_px':     1000,
            'min_overlap_px':  200,
        }],
        condition=IfCondition(use_row_counter)           
    )

    rqt_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_overlay',
        arguments=['--force-discover'],
        output='screen',
        condition=IfCondition(use_rqt)                   
    )

    return LaunchDescription([
        use_row_counter_arg,
        use_rqt_arg,
        use_gpu_arg,
        detector_node,
        row_counter_node,
        rqt_view
    ])
