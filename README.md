# BAMBI Detection

The [BAMBI project](https://www.bambi.eco/) uses camera drones together with artificial intelligence to automatically monitor wildlife. 
Light field technology is used, which for the first time makes it possible to visualize what is happening on the forest floor, and thus to detect animals with a high degree of reliability. 
Based on this technology, an AI-powered system can detect and automatically classify animals on the forest floor and in the open terrain, thus allowing an area-wide and accurate count of wildlife that has not been possible until now.

## Setup

This repository is using code from the [alfs_py project](https://github.com/bambi-eco/alfs_py).

So activate your environment (or do it globally) and install alfs_py since it is not available on pypi:

```
pip install <path-to-alfs_py-project>
```

Afterwards install the missing dependencies using pip

```
pip install -r requirements.txt 
```

Currently only supports Ultralytics models that are hosted in our [Models repository](https://github.com/bambi-eco/Models).
So make sure these models are available as environment variable (c.f. repository readme).

## Usage

Use the provided 'bambi_detection.py' script and provide the required input data.
Afterwards just hit run ;)