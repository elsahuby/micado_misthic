from setuptools import find_packages, setup


setup(name='micado_misthic',
      version='1.0',
      description='MISTHIC, MIcado SimulaTor for HIgh Contrast: image simulator for the high contrast mode of MICADO, and realted functions',
      url='https://gitlab.obspm.fr/ehuby/micado_misthic.git',
      author='E. Huby, P. Baudoz',
      author_email='elsa.huby@obspm.fr',
      license='',
      packages=find_packages(),
      install_requires=[
      	'numpy >= 1.26.4',
        'configobj',
        'validator', 
        'opencv-python',
        #'os',
        #'inspect',
        'Astropy >= 6.0.1',
        'scipy',
        'matplotlib',
        'PyQt6',
        #'pylab-sdk',
        'pyfftw',
        'poppy', #==0.9.1',
        'tqdm', 
        'pandas'
      ],
      include_package_data = True,
      package_data={'micado_misthic.perfCalculator.interface_modules': ['*.md']},
      zip_safe=False)
