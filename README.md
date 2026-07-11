<table border="0">
<tr>

<td border="0" width="110" valign="middle">
<img src="DemoImg/MedXina.jpg" width="90">
</td>

<td border="0">

<h1 style="margin:0;">MRI Learning Studio</h1>

<p style="margin-top:8px;">
<b>Interactive MRI Physics Simulator for Medical Education</b>
</p>

</td>

</tr>
</table>

---

MRI Learning Studio is an interactive educational platform designed to help students understand MRI physics through real-time visualization and simulation.

Each lesson demonstrates a specific MRI concept using interactive 3D animations and parameter controls.

## Features

- Interactive MRI physics demonstrations
- Real-time 3D visualization
- Step-by-step lessons
- Easy-to-use graphical interface (Tkinter)
- Modular lesson structure for future expansion

## Project Structure

```text
MRI-Learning-Studio/
│
├── run.py                # Launch the lesson selector
├── lessons/              # Individual lesson scripts
├── DemoImg/              # Images and logo
└── README.md
```

## Getting Started

1. Clone this repository.

```bash
git clone https://github.com/yourname/MRI-Learning-Studio.git
```

2. Install the required packages.

```bash
pip install numpy matplotlib
```

3. Launch the application.

```bash
python run.py
```

## Using the Lesson Launcher

After launching `run.py`, a graphical interface will appear.

Simply:

1. Select a lesson from the list.
2. Click **Launch Lesson**.
3. Explore the interactive MRI simulation.

Future lessons can be added by simply placing new Python scripts inside the `lessons/` folder.

## Current Lessons

- 01 – MRI Fundamentals: Coils and Magnetization

More lessons will be added in future updates.

---

Created for interactive MRI education.
