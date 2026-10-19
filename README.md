# The Southern Ocean CO2 sink in an evolving observing system

Louise Delaigue¹, Peter Landschützer², Cathy Wimart-Rousseau³, Lisandro A. Arbilla⁴˒⁵˒⁶, Jean-Baptiste Sallée⁴, Seth Bushinsky⁷, Hervé Claustre¹ and Raphaëlle Sauzède⁸

<details>
<summary>Author affiliations</summary>

1. Sorbonne Université, CNRS, Laboratoire d'Océanographie de Villefranche, Villefranche-sur-Mer, France
2. Flanders Marine Institute (VLIZ), Ostend, Belgium
3. National Oceanography Centre, Southampton, UK
4. LOCEAN, Sorbonne Université, CNRS, IRD, MNHN, LOCEAN/IPSL, Paris, France
5. Instituto Argentino de Oceanografía (IADO, CONICET-UNS), Bahía Blanca, Argentina
6. Departamento de Geografía y Turismo, Universidad Nacional del Sur (UNS), Bahía Blanca, Argentina
7. Department of Oceanography, SOEST, University of Hawai'i at Mānoa, Honolulu, HI, USA
8. Sorbonne Université, CNRS, Institut de la Mer de Villefranche, Villefranche-sur-Mer, France

</details>

Corresponding author: Louise Delaigue ([louise.delaigue@imev-mer.fr](mailto:louise.delaigue@imev-mer.fr))

> [!IMPORTANT]
> This study is currently under review for publication in *Global Biogeochemical Cycles*.

## Abstract

The Southern Ocean is a major sink for atmospheric CO2, yet the magnitude and trend of this sink remain uncertain due to sparse and seasonally biased observations. Reconstructions based on the Surface Ocean CO2 Atlas (SOCAT) indicate a strengthening of this sink since the early 2000s, but such signals may reflect evolving observational coverage rather than basin-scale change. Here, we combine SOCAT with partial pressure of CO2 (pCO2) estimates from Biogeochemical (BGC-) Argo floats to assess how expanded autonomous sampling alters estimates of Southern Ocean air–sea CO2 exchange. Using feedforward multilayer perceptron models trained on SOCAT-only, BGC-Argo-only, and a merged, uncertainty-weighted combination, we compute monthly air–sea CO2 fluxes south of 30°S for 2004–2024. Ship-only observations imply a strong intensification of the CO2 sink, but this signal disappears once floats are included, yielding a combined reconstruction with a near-zero, non-significant long-term trend. The uncertainty-weighted combined reconstruction yields a mean Southern Ocean CO2 uptake of −1.20 ± 0.13 Pg C yr⁻¹ over this period, ~0.2 Pg C yr⁻¹ weaker than the fCO2-based Global Carbon Budget ensemble mean, as expected from the summer-biased ship sampling. This estimate lies within the spread of atmospheric inversion and global ocean biogeochemical models, while limiting reported systematic biases in float-derived pCO2 by anchoring autonomous observations to ship-based measurements. Integrating complementary observation streams provides a benchmark for Southern Ocean CO2 uptake whose differences from other products can be interpreted in terms of known physical and biogeochemical processes, illustrating how treatment of sampling biases can reconcile ship- and float-derived constraints.

## Repository contents

This repository contains the scripts and notebooks used to process the data, train the multilayer perceptron (MLP) reconstructions, generate predictions, compute air–sea CO2 fluxes, and produce all figures and tables in the manuscript and Supplementary Material.

| Folder / file | Contents |
|---|---|
| `MODEL_SOCAT_ONLY_OBS_v2/` | SOCAT-only pipeline: preprocessing, data split, architecture search, Monte Carlo ensemble, independent validation |
| `MODEL_FLOATS_v2/` | BGC-Argo float-only pipeline (same structure) |
| `MODEL_FLOATS_SUMMER_ONLY_v2/` | Summer-only float sensitivity experiment (Fig. S16) |
| `MODEL_SOCAT_AND_FLOATS_v2/` | Combined, uncertainty-weighted SOCAT + BGC-Argo pipeline |
| `PREDICTION_MATRIX_v2/` | Construction of the gridded predictor dataset (1° × 1°, monthly, south of 30°S) and ensemble pCO2 predictions |
| `Figure1.ipynb` | Fig. 1 |
| `Figures3_4_5_6_7_Table2_Table3_documented_si10.ipynb` | Flux calculation, Figs. 3–7, Tables 1–2 and Tables S2–S4 |
| `FigureS10.ipynb`, `Supp_figure_*.ipynb` | Supplementary figures |
| `figs/` | Output figures |

## Data availability

> [!NOTE]
> Input datasets and large intermediate files (gridded predictor fields, training tables and pCO2 prediction datasets, several GB in total) are **not included in this repository** because of GitHub file-size limits. All input datasets are open access and can be downloaded from the sources listed below. The processed intermediate files and the final reconstructed pCO2 and flux products are available from the corresponding author upon request.

### Observations

| Dataset | Source |
|---|---|
| Surface Ocean CO2 Atlas, version 2025 (SOCATv2025), Southern Ocean regional synthesis file | Bakker et al. (2016, 2025); [socat.info](https://socat.info) |
| BGC-Argo float carbonate-system estimates (CANYON-B / CONTENT) | Copernicus Marine Service, `MULTIOBS_GLO_BGC_NUTRIENTS_CARBON_PROFILES_MYNRT_015_009`; [marine.copernicus.eu](https://marine.copernicus.eu) |

BGC-Argo data were collected and made freely available by the International Argo Program and the national programs that contribute to it ([argo.ucsd.edu](https://argo.ucsd.edu), [ocean-ops.org](https://www.ocean-ops.org)).

### Predictor fields

| Variable | Source |
|---|---|
| Temperature, salinity, mixed layer depth | ARMOR3D, Copernicus Marine Service, `MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012` |
| Sea level anomaly | DUACS, Copernicus Marine Service, `SEALEVEL_GLO_PHY_L4_MY_008_047` |
| Remote sensing reflectance | Copernicus Marine Service, `OCEANCOLOUR_GLO_BGC_L3_MY_009_103` |
| Photosynthetically available radiation | GlobColour, [hermes.acri.fr](https://hermes.acri.fr) |
| 10 m winds (u10, v10, si10) and mean sea-level pressure | ERA5 monthly averaged data on single levels, Copernicus Climate Data Store ([doi:10.24381/cds.f17050d7](https://doi.org/10.24381/cds.f17050d7)) |
| Atmospheric xCO2 | Cape Grim Baseline Air Pollution Station (CSIRO / Australian Bureau of Meteorology) |
| Bathymetry | GEBCO 2024 Grid ([doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f](https://doi.org/10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f)) |

### Flux calculation and benchmarks

| Dataset | Source |
|---|---|
| Global Carbon Budget 2025 fCO2 products, GOBMs and atmospheric inversions | Friedlingstein et al. (2026); [globalcarbonbudget.org](https://globalcarbonbudget.org) |
| Ocean area south of 30°S | GOAS v1 polygon dataset, Flanders Marine Institute (2021); [marineregions.org](https://www.marineregions.org) |
| Sea ice concentration climatology (1991–2020) | Danish Meteorological Institute from OSI SAF data; [ocean.dmi.dk](https://ocean.dmi.dk/arctic/) |
| Polar Front position | Park & Durand (2019), SEANOE ([doi:10.17882/59800](https://doi.org/10.17882/59800)) |

To rerun the pipelines, download the datasets above into a `data/` folder at the repository root. The expected file paths are set at the top of each script.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
