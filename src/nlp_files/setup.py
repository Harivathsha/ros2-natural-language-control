from setuptools import find_packages, setup

setup(
    name='nlp_files', version='0.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/nlp_files']),
                ('share/nlp_files', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    description='Learned natural language control for a ROS 2 simulation',
    entry_points={'console_scripts': ['slot_dataset = nlp_files.data.slot_dataset:main', 'train_semantic_model = nlp_files.training.train_semantic_model:main', 'nlp_pipeline = nlp_files.core.nlp_pipeline:main', 'nlp_ros_node = nlp_files.ros_nodes.nlp_ros_node:main', 'ros_command_client = nlp_files.ros_nodes.ros_command_client:main', 'robot_ui = nlp_files.ros_nodes.robot_ui:main', 'motion_handler = nlp_files.ros_nodes.motion_handler:main', 'navigation_handler = nlp_files.ros_nodes.navigation_handler:main', 'diagnosis_handler = nlp_files.ros_nodes.diagnosis_handler:main', 'safety_rule_handler = nlp_files.ros_nodes.safety_rule_handler:main']},
)
