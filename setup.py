from setuptools import setup, find_packages

setup(
    name='mc-gridworld',
    version='1.0',
    packages=find_packages(include=['envs', 'agents']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'gym'
    ],
    license='MIT',
    author='pmitseas',
    author_email='petros_94@icloud.com',
    description='An implementation of the Monte Carlo Reinforcement Learning algorithm on the Gridworld environment'
)
