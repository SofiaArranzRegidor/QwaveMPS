import setuptools

setuptools.setup(
    name="QwaveMPS",
    version="1.0.0", 
    author="Sofia Arranz Regidor",
    author_email="18sar4@queensu.ca",
    description="Package to solve waveguide QED problems using MPS",
    url="https://github.com/SofiaArranzRegidor/QwaveMPS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    license="GNU General Public License v3 (GPLv3)",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),    
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib']
)
