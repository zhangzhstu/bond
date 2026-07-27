# BOND — project page

Source for <https://zhangzhstu.github.io/bond/>, the page for

> **What the Kernel Measures: Bounding Feature Geometry for Few-Shot Molecular
> Property Prediction.** Zihang Zhang, Jiayi Li, Yanan Sun, Jiujun Cheng,
> Zhenyu Lei, Shangce Gao. KDD 2027.

## What is here

    index.html        the whole page — one file, no build step

The paper PDF is deliberately NOT in this folder. An unlinked file in a public
repository is still public — it is served at its own URL and indexed — so while
the paper button reads TBD the PDF is kept out of the repository entirely. It
lives in `../bond-site-assets-heldback/`; drop it back in beside `index.html`
and point the button at it when the paper is ready to be public.

`index.html` has no dependencies except one Google Fonts stylesheet for the
handwriting face. Everything else — the sketched frames, the arrows, the live
figure — is inline SVG and about 300 lines of plain JavaScript. Open the file
directly in a browser and it works; there is nothing to install and nothing to
compile.

## Putting it online

The URL in the paper is `https://zhangzhstu.github.io/bond/`, so the repository has
to be named **`bond`** under the GitHub account **`zhangzhstu`**. A different
repository name gives a different URL and the link in the abstract stops
resolving.

    cd bond-site
    git init -b main
    git add .
    git commit -m "BOND project page"
    git remote add origin https://github.com/zhangzhstu/bond.git
    git push -u origin main

Then, in the repository on github.com: **Settings → Pages → Build and
deployment → Source: Deploy from a branch**, branch `main`, folder `/ (root)`.
Save. The first build takes a minute or two; after that every push republishes.

There is no Jekyll front-matter and no `_config.yml`, so nothing is processed —
GitHub serves `index.html` as it is. If you ever add a directory beginning with
an underscore, add an empty `.nojekyll` file at the root so Jekyll does not skip
it.

## The interactive figure

The middle panel is the paper's Figure 2, live. Nine fixed points, one slider
that scales the feature map, and the Matérn-5/2 kernel those points produce:

- **collapsed** — every pairwise distance goes to zero, every entry of K goes to
  one, and the effective rank falls to 1. The length-scale is invisible.
- **median heuristic** — the initialisation the deep-kernel literature uses puts
  the median pair at r = √2, which keeps 94% of the available Fisher
  information.
- **dispersed** — every distance runs away, K goes to the identity, effective
  rank 9, and the length-scale is invisible again for the opposite reason.

The numbers under the slider are computed, not hard-coded: `matern52`, `g(r) =
−r·k′(r)`, and the effective rank as `(Σλ)²/Σλ² = tr(K)²/‖K‖²_F`. The
"information kept" readout is the ratio of `g²`, not of `g`, because the Fisher
information for log ℓ is quadratic in `g` — the same distinction the paper draws
in Corollary 4.5.

## Editing

Everything lives in `index.html`:

- **colours** — the `:root` block at the top; there is a dark-mode override at
  the bottom of the stylesheet.
- **hand-drawn look** — the two SVG filters at the top of `<body>`, plus
  `wobble()` and `wobbleRect()` in the script. Straight lines read as
  machine-drawn however much you filter them, so the paths wander off the ideal
  line before the filter is applied.
- **the figure** — `PTS` holds the nine points; they are written out rather than
  generated so every reader sees the same picture as the paper.
