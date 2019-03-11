import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()
with open('requirements.txt', 'r') as f:
    requirements = f.read().strip('\n').split('\n')

entry_points = {
    'console_scripts': [
        'compute_eis_pointing = eis_pointing.driver_cli:main',
        ]
    }

package_data = {
    '': ['*.pro'],
    }

setuptools.setup(
    name='eis_pointing',
    version='2019.03.11',
    author='Gabriel Pelouze',
    author_email='gabriel.pelouze@ias.u-psud.fr',
    description='Tools to correct the pointing of Hinode/EIS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gpelouze/eis_pointing',
    entry_points=entry_points,
    package_data=package_data,
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
