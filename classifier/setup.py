from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras>=2.0.8',
					 # 'gcloud>=0.18'
					 # 'cloudstorage>=0.4'
					 # 'google-cloud-storage>=1.4.0',
					 'scikit-learn>=0.18.0',
					 'Pillow>=4.2.0',
                     'h5py>=2.7.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)