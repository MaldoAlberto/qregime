# Dissemination kit

Drafts and a checklist for launching `qregime`. Edit freely; the numbers are
the ones in the paper and README, so keep them in sync if they change.

## Before announcing (in this order)

1. **Check with your employer before publishing,** if you have one. Many
   employment agreements cover code and publications in the same field as
   your job, and require approval for open-source releases or for listing
   the work as independent.
2. **Push and check that CI is green** (Actions tab). The badge in the README
   stays grey until the first run.
3. **Repository settings:** add a one-line description and topics
   (`quantum-computing`, `quantum-error-correction`, `error-mitigation`,
   `fault-tolerance`, `zx-calculus`, `qiskit`, `resource-estimation`), and
   upload `docs/budget_staircase.png` as the social preview image
   (Settings > General > Social preview) so links show the chart.
4. **Open 4-5 issues labelled `good first issue`** from the table in
   `CONTRIBUTING.md`. A repo with visible, scoped entry points gets contributors;
   one without them gets stars at most.
5. **Tag a release `v0.1.0`.** If you connect the repo to Zenodo first
   (zenodo.org > GitHub), the release gets a citable DOI automatically; add it
   to `CITATION.cff` and the README.
6. **Publish to PyPI** (the name `qregime` was free on 29 Sep 2026):
   `python -m build && twine upload dist/*`. Then the quickstart becomes
   `pip install qregime`.
7. **Post the paper to arXiv (quant-ph)** if possible. First-time submitters
   need an endorsement from someone who already publishes in that category.
   Link the arXiv version everywhere once it exists.

## Where to post

| Channel | Why | Tone |
|---|---|---|
| Unitary Foundation Discord | They fund open-source quantum tools (microgrants, Deltakit Community Fund) and run the Metriq benchmark this project was inspired by | technical, mention the fit with their work |
| Qiskit community channels | The tool is built on Qiskit and ingests Qiskit's QASM | technical, show the CLI |
| r/QuantumComputing | Broad technical audience | lead with the chart and the "GAP" idea |
| LinkedIn | Industry and hiring managers | lead with the decision problem |
| X / Bluesky / Mastodon | Researchers | short thread, one number per post |
| Curated lists (e.g. awesome-quantum-software on GitHub) | Long-term discoverability | open a PR adding one line |

Post in one or two places first, answer every question for a day, then widen.
Early, thoughtful replies matter more than reach.

## LinkedIn / long post (English)

> Before you run a quantum circuit, someone has to decide how errors will be
> handled: mitigate them on raw qubits, correct them with a small code, or pay
> for full fault tolerance. That choice is the difference between 16 physical
> qubits and 30,000, and it is usually made by rule of thumb.
>
> I've open-sourced **qregime**, a tool that makes the decision explicit: give
> it an OpenQASM circuit and a hardware profile, and it tells you which
> circuit-reduction techniques are worth trying, what full fault tolerance
> would cost, and which regime is the cheapest that meets your reliability
> target. When none does, it says how large the gap is, and how much hardware
> would close it.
>
> A few things I learned building it:
> - For one Shor-like circuit, partial error correction reaches 68% reliability
>   at ~9,700 physical qubits, a third of the cost of full fault tolerance.
> - On 99.99%-fidelity hardware, the same circuit needs no error correction at all.
> - ZX-calculus provably cannot reduce T-count on QAOA-style cost layers (a
>   short proof is in the paper).
> - Honest negative result: no cheap structural predictor of "will ZX-calculus
>   help?" generalizes to new circuit families. My own first number (82.9%)
>   didn't survive stricter validation and I retracted it.
>
> Everything is reproducible: code, data, tests, an executed tutorial and the
> paper. Contributions very welcome, especially new hardware profiles.
>
> [link]

## Publicación larga / LinkedIn (español)

> Antes de correr un circuito cuántico hay que decidir cómo manejar los
> errores: mitigarlos sobre los qubits físicos, corregirlos con un código
> pequeño, o pagar la tolerancia a fallas completa. Es la diferencia entre 16
> qubits físicos y 30.000, y casi siempre se decide a ojo.
>
> Liberé como código abierto **qregime**, una herramienta que vuelve explícita
> esa decisión: le das un circuito OpenQASM y un perfil de hardware, y te dice
> qué técnicas de reducción vale la pena probar, cuánto costaría la tolerancia
> a fallas completa, y cuál es el régimen más barato que cumple tu objetivo de
> confiabilidad. Si ninguno lo cumple, te dice de qué tamaño es la brecha y
> cuánto hardware haría falta para cerrarla.
>
> Algunas cosas que aprendí en el camino:
> - Para un circuito tipo Shor, la corrección de errores parcial llega a 68% de
>   confiabilidad con ~9.700 qubits físicos: un tercio del costo de la
>   tolerancia a fallas completa.
> - En hardware con 99,99% de fidelidad, el mismo circuito no necesita
>   corrección de errores.
> - ZX-calculus está demostrado que no puede reducir el T-count en capas de
>   costo tipo QAOA (la demostración está en el paper).
> - Resultado negativo, honesto: ningún predictor estructural barato de "¿ayudará
>   ZX-calculus?" generaliza a familias de circuitos nuevas. Mi primera cifra
>   (82,9%) no sobrevivió una validación más estricta y la retracté.
>
> Todo es reproducible: código, datos, tests, un tutorial ejecutado y el paper.
> Contribuciones bienvenidas, sobre todo perfiles de hardware nuevos.
>
> [link]

## Short thread (X / Bluesky / Mastodon)

1. QEM, QEC or FTQC? The answer decides whether a circuit needs 16 physical
   qubits or 30,000. I built an open tool that makes that decision explicit
   and reproducible from a QASM file. [link + chart]
2. It profiles the circuit, says which reductions are worth trying, estimates
   full fault-tolerance cost from its own Stim+PyMatching threshold fit, and
   picks the cheapest regime that meets your reliability target.
3. When no regime works, it doesn't force an answer: it reports the gap, and
   the best reliability your qubit budget can buy (the staircase in the chart).
   A proof in the paper guarantees that curve never goes down.
4. Partial QEC gets a Shor-like circuit to 68% reliability at 1/3 the cost of
   full fault tolerance. On 99.99% hardware it needs no QEC at all.
5. Negative results are in there too: no cheap structural predictor tells you
   whether ZX-calculus will help on a new circuit family. I retracted my own
   82.9% when it didn't survive leave-one-family-out validation.
6. Apache 2.0, with tests, an executed tutorial and the full research log.
   Good first issues are open; hardware profiles especially welcome. [link]

## Reddit (r/QuantumComputing)

**Title:** I open-sourced a tool that decides QEM vs partial QEC vs full fault
tolerance for a given circuit, and reports the gap when nothing works

**Body:** lead with the chart and two sentences on the problem, then the
quickstart, then the "what didn't work" list from the README. Ask a concrete
question to invite discussion, for example: "Which hardware profiles would you
want to see added, and where do you get reliable two-qubit error rates?"
