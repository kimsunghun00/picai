from setuptools import setup, find_packages

if __name__ == '__main__':
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    setup(
        name='sunghun_lib',
        version='0.0.1',
        desrciption='sunghun pip install test'
        url='https://github.com/kimsunghun00/picai.git'
        author='sunghun'
        author_meail='uiop8533@gmail.com',
        python_requires='>=3.7',
        install_requires=requirements
    )

