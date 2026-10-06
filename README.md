# Nationwide landslide susceptibility of Kyrgyzstan (250 m)

This repository holds the Google Earth Engine and Python code for:

> Duulatov E. Nationwide landslide susceptibility of Kyrgyzstan: a statistically tested comparison of six machine-learning models with uncertainty and consensus mapping. *Landslides* (submitted).

**Method:** 16 conditioning factors on a 250 m grid; 5,143 landslide and 5,143 non-landslide cells; six tuned models (Random Forest, gradient boosting, XGBoost, SVM, deep neural network, k-nearest neighbours).

**Evaluation:**
- random and 50 km spatial-block cross-validation;
- DeLong, Friedman and Wilcoxon tests;
- SHAP explanations;
- sensitivity analyses;
- ensemble, uncertainty and consensus maps;
- 2025 population exposure.

**Data:** the predictor layers, model outputs, trained models and result tables are on Zenodo: [https://doi.org/10.5281/zenodo.23192152](https://doi.org/10.5281/zenodo.23192152).

## Repository layout

```
gee/kg_factors_250m.js     Earth Engine script that derives the conditioning factors and exports them as GeoTIFFs
python/                    Sampling, modelling, validation, SHAP, sensitivity analyses, maps and figures
python/verify_deposit.py   Checks a downloaded copy of the Zenodo record against the published results
docs/data_sources.md       Source datasets, licences and citations
requirements.txt           Exact Python package versions (Python 3.11)
environment.yml            The same environment for conda
```

## Quick check (about one minute, no GIS inputs needed)

```bash
pip install -r requirements.txt
# download the Zenodo record into ./zenodo
python python/verify_deposit.py zenodo
```

This applies the six trained models to the sample table. It reproduces the published test-set probabilities and test AUCs (RF 0.9595, GBT 0.9574, XGBoost 0.9613, SVM 0.9516, DNN 0.9465, KNN 0.9432).

## Full pipeline

Run the Python scripts from inside `python/`. Intermediate files go to `python/data/`, `python/raw250/`, `python/out/` and `python/rasters/`; Git ignores these folders.

### Inputs not in this repository

Put these in `python/aux/`, or point the environment variable `KG_AUX` to another folder.

| Path | Content | Source |
|---|---|---|
| `kg_adm1/kg_adm1_9.shp` | Oblast boundaries (fields ADM1_EN, oblast_id) | Ministry of Emergency Situations KR (2018), via HDX `cod-ab-kgz` |
| `adm2/adm2.shp` | Raion boundaries (statistics only) | Ministry of Emergency Situations KR (2018), via HDX `cod-ab-kgz` |
| `lakes/lakes_reserv_main.shp` | Six largest lakes and reservoirs (masked) | [VERIFY: source] |
| `faults/TS_Active_Flts_update.shp` | Active-fault traces (UTM 43N) | [Active Tectonics of the Northern Tien Shan](http://activetectonics.asu.edu/N_tien_shan/N_tien_shan_data.html), Arizona State University |
| `geology/geology.shp` | Geological formation map | Layer `geonode:geology`, [GeoNode of the Water Resources Service](https://geonode.water.gov.kg), Kyrgyz Republic |
| `cca/inform_cca_admin1_gadm.shp` | Country outlines (Fig. 1 locator only) | OCHA ROCCA / INFORM Central Asia admin 1 (GADM) |
| `python/data/ls_new_xy.npy` | Landslide points (x, y, EPSG:32643) | Compiled inventory; available on request (see the paper) |

### Run order

| Step | Script(s) | Output |
|---|---|---|
| 1 | `gee/kg_factors_250m.js` (Earth Engine Code Editor; first upload `kg_adm1_9.shp` as a table asset and set its path at the top of the script) | Six int16 GeoTIFF exports to the Google Drive folder `kg_landslide_gee` |
| 2 | `mosaic.py` | Tiles placed in `raw250/tiles/` are mosaicked into `raw250/` |
| 3 | `gen_samples.py` | `data/samples.csv`: landslide cells, random non-landslide cells at least 1 km away, 70/30 split |
| 4 | `ml.py` | Grid-search tuning, test metrics, national probability grids, `data/ml_models.joblib` |
| 5 | `ml_spatial.py`, `extra_cvtrain.py` | Random and spatial-block 5 × 5 CV with Friedman and Wilcoxon tests |
| 6 | `shap_run.py`, `shap_all.py`, `shap_all_fig.py` | SHAP: detailed for RF and XGBoost, comparable for all six models |
| 7 | `local_stats.py`, `extra_rates.py`, `extra_sens.py` | Frequency ratio, VIF and class statistics; success and prediction rates; classification schemes; absence-sampling, ablation and leave-one-oblast-out tests |
| 8 | `export_rasters.py`, `export_predictors.py` | GeoTIFFs of the model outputs and predictors (as on Zenodo) |
| 9 | `pop_exposure.py`, `raion_stats.py` | Population exposure (needs `raw250/kg250_pop2025b.tif`, see below) and raion statistics |
| 10 | `ml_figs.py`, `make_maps.py`, `make_maps2.py`, `make_charts.py`, `make_flow.py`, `make_extra_figs.py`, `pop_figs.py` | Figures, DeLong tests and supplementary figures |
| 11 | `map01_inset.py`, `fig1_cvd.py`, `roc_cvd.py`, `fig6_nocap.py`, `fig8_cvd.py`, `strip_titles.py`, `combine_fig7.py`, `build_cvd_final.py` | Final main figures (colour-blind-safe, no in-figure titles) |
| 12 | `cvd_sim.py` | Previews under simulated deuteranopia, protanopia and tritanopia |

`common.py` defines the grid and data loaders; `mapkit.py` holds the map helpers. All models use `random_state = 42`.

The 2025 population grids (WorldPop R2025A, GHS-POP R2023A) were summed to the 250 m grid in Earth Engine and exported as `raw250/kg250_pop2025b.tif`. [PENDING: add that Earth Engine snippet as `gee/kg_population_2025.js`.]

## Citation

If you use this code, please cite the paper and the software record (see `CITATION.cff`). If you use the data, also cite the Zenodo data record.

## Licence

The code is released under the MIT licence (see `LICENSE`). The data on Zenodo carry their own licences: CC BY 4.0, and CC BY-NC 4.0 for the MERIT Hydro-derived layers.
