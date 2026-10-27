# Office demo scene

Use the [Getting Started walkthrough](../../docs/GETTING_STARTED.md) to reconstruct
this office from the included video through final scene delivery.

![Whitebox render of the office reconstruction](../../docs/media/office48_whitebox.jpg)

A home office with a vaulted, beamed ceiling, reconstructed by the agentic workflow
from a 10-second video. It is the office example on the
[project page](https://kevinxu02.github.io/real2sim-indoor-site/).

`whitebox.blend` (Blender 5.2) contains the editable room and the camera
recovered from the video:

- 401 objects: walls, beamed ceiling, windows, glazed door, trestle desk,
  swivel and guest chairs, ladder shelf, side cabinet, chandelier and props.
  Objects are named by role, such as `furniture/...` and `architecture/...`.
- The source camera animated over frames 1–303 at 30 fps, 1280×720.
- One neutral material; the file is self-contained.

## Inspect, edit and render

Ask the agent to [show this example in the Gallery](../../docs/GETTING_STARTED.md#1-explore-the-supplied-result).
You can also view the [supplied render](../../docs/media/office48_whitebox.mp4)
directly. Open Blender when you want to edit the scene:

```bash
blender examples/office48/whitebox.blend
```

To render a single frame without the UI (frame 152 here), writing outside the
repository:

```bash
blender --background examples/office48/whitebox.blend \
  -o /tmp/office48/frame_#### -F PNG -f 152
```

Replace `-f 152` with `-a` to render the full shot with Cycles.

