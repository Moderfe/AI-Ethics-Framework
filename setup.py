from setuptools import setup, find_packages

setup(
    name=\'ai_ethics_framework\',
    version=\'0.1.0\',
    packages=find_packages(where=\'src\'),
    package_dir={\'\': \'src\'},
    install_requires=[
        \'pandas>=1.0.0\',
        \'scikit-learn>=0.24.0\',
        \'numpy>=1.18.0\',
    ],
    author=\'Marcus D. Sterling\',
    author_email=\'erarealorgrun@gmx.net\',
    description=\'A comprehensive framework for developing, evaluating, and deploying AI systems responsibly and ethically.\',
    long_description=open(\'README.md\').read(),
    long_description_content_type=\'text/markdown\',
    url=\'https://github.com/Moderfe/AI-Ethics-Framework\',
    classifiers=[
        \'Programming Language :: Python :: 3\',
        \'License :: OSI Approved :: MIT License\',
        \'Operating System :: OS Independent\',
        \'Intended Audience :: Developers\',
        \'Intended Audience :: Science/Research\',
        \'Topic :: Scientific/Engineering :: Artificial Intelligence\',
    ],
    python_requires=\'>=3.8\',
)
