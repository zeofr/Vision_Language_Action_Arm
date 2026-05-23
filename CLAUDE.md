# VLA Robotic Arm

- Be brief. Return only the answer, steps, or code. No recap unless asked.
- Treat `Team_HARDWARE_EMBEDDED_WORKPLAN.md` and `VLA_Robotic_Arm_Project_Report_FINAL.md` as source docs. Read only when needed.

## Graphify

- For repo structure, dependencies, architecture, impact, or "where is X / what uses X" questions, use Graphify first.
- First read `graphify-out/GRAPH_REPORT.md` if it exists.
- Then prefer:
  - `graphify query "<question>" --budget 1200`
  - `graphify path "<node A>" "<node B>"`
  - `graphify explain "<node>"`
- Do not use `./graphify`; use `graphify`.
- Do not start with `grep`/`rg` for understanding architecture or dependencies.
- Use raw search only after Graphify narrows the area, or when exact code/file matches are required.

## Hardware Constraints

- Target: Teensy 4.1, PlatformIO/C++, strict 50 Hz control loop.
- Never suggest powering servos and Raspberry Pi from the same unisolated 12V rail.

## Instruction Style

- When giving procedural guidance, write it as a numbered step-by-step lab manual.
- Include each click, command, wiring action, value to enter, and the expected result after each step.
- Prefer completeness over brevity for procedures, but stay concise elsewhere.
- Expand only the steps the user should perform now; do not add background unless needed to complete the step correctly.
