# Gazetteer

`venus.geojson` is not committed. Fetch the IAU nomenclature for Venus from
<https://planetarynames.wr.usgs.gov/> (Advanced Search → Target: Venus → GeoJSON) and
save it here as `venus.geojson`.

Longitudes are published 0–360 east; `src/features/gazetteer.ts` converts to Cesium's
−180…180 in exactly one place (`toCesiumLongitude`). Do not convert anywhere else.

Until the file exists the search box falls back to the hard-coded `SITES` list in
`src/venus.ts`, which is enough to demo the fly-to.
