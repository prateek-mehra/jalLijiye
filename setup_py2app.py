from setuptools import setup


APP = ["jallijiye_app.py"]
DATA_FILES = [
    ("config", ["config/defaults.yaml"]),
    ("models/bottle_v3/weights", ["models/bottle_v3/weights/best.pt"]),
    ("", ["yolov8n.pt"]),
]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": None,
    "plist": {
        "CFBundleName": "JalLijiye",
        "CFBundleDisplayName": "JalLijiye",
        "CFBundleIdentifier": "com.prateek.jallijiye",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSUIElement": True,
        "NSCameraUsageDescription": "JalLijiye uses the camera to detect bottle drinking events on-device.",
    },
    "packages": ["app", "rumps", "yaml"],
    "includes": [
        "Foundation",
        "AppKit",
    ],
    "site_packages": True,
}


setup(
    app=APP,
    name="JalLijiye",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
