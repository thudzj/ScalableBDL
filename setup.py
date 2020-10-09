from setuptools import setup, find_packages

install_requires = [
    'torch>=1.3.0',
]

setup(
    name='scalablebdl',
    version='0.0.1',
    description='Package for scalable Bayesian deep learning',
    url='git@github.com:thudzj/ScalableBDL.git',
    author='Zhijie Deng',
    author_email='dzj17@mails.tsinghua.edu.cn',
    license='license.txt',
    packages=find_packages(),
    include_package_data = True,
    platforms = "any",
    zip_safe=False,
    package_data={
        'scalablebdl': [
            'mean_field/*'
        ],
    },

)
