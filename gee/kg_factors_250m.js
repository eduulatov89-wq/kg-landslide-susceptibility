/**************************************************************************************
 * Kyrgyzstan landslide susceptibility - conditioning factors at 250 m (full precision)
 * Grid: EPSG:32643, 250 m, origin (3500, 4796000), 3736 x 1844 cells.
 * Terrain derivatives are computed on a 50 m DEM and aggregated exactly (5 x 5) to 250 m.
 * Six GeoTIFF exports (int16, nodata -32768) to Drive folder kg_landslide_gee:
 *   kg250_terrainA  elevation (m), slope (deg x10), aspect (deg; -1 flat)
 *   kg250_terrainB  plan_curv, profile_curv (1/m x 1e5)
 *   kg250_terrainC  relief (m, max-min of 50 m cells), twi (x100)
 *   kg250_envA      rain_annual (mm), rain_wetq (mm), lulc (WorldCover code), oblast (id)
 *   kg250_envB      dist_roads (m, cap 32000), dist_rivers (m, cap 25600)
 *   kg250_envC      ndvi (x1e4), clay (g/kg), sand (g/kg)
 **************************************************************************************/
var FOLDER = 'kg_landslide_gee';
// Oblast boundaries (kg_adm1_9.shp, see README): upload as an Earth Engine table asset and set its path here.
var adm = ee.FeatureCollection('projects/YOUR_EE_PROJECT/assets/kg_adm1_9');
var CRS = 'EPSG:32643';
var T250 = [250, 0, 3500, 0, -250, 4796000];
var T50  = [50, 0, 3500, 0, -50, 4796000];
var T100 = [100, 0, 3500, 0, -100, 4796000];
var REGION = ee.Geometry.Rectangle([3500, 4335000, 937500, 4796000], CRS, false);
var AOI_LL = ee.Geometry.Rectangle([69.0, 39.0, 80.5, 43.5]);
var NODATA = -32768;
function to250(img, reducer, maxPixels) {
  return img.reduceResolution({reducer: reducer, maxPixels: maxPixels || 64})
            .reproject({crs: CRS, crsTransform: T250});
}

// ---------------- DEM at 50 m and derivatives
var glo = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(AOI_LL).select('DEM');
var dem30 = glo.mosaic().setDefaultProjection(glo.first().projection());
var dem = dem30.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64})
               .reproject({crs: CRS, crsTransform: T50});
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);
var L = 50;
var p   = dem.convolve(ee.Kernel.fixed(3, 3, [[0,0,0],[-1,0,1],[0,0,0]], -1, -1, false)).divide(2 * L);
var q   = dem.convolve(ee.Kernel.fixed(3, 3, [[0,1,0],[0,0,0],[0,-1,0]], -1, -1, false)).divide(2 * L);
var zxx = dem.convolve(ee.Kernel.fixed(3, 3, [[0,0,0],[1,-2,1],[0,0,0]], -1, -1, false)).divide(L * L);
var zyy = dem.convolve(ee.Kernel.fixed(3, 3, [[0,1,0],[0,-2,0],[0,1,0]], -1, -1, false)).divide(L * L);
var zxy = dem.convolve(ee.Kernel.fixed(3, 3, [[-1,0,1],[0,0,0],[1,0,-1]], -1, -1, false)).divide(4 * L * L);
var g2 = p.pow(2).add(q.pow(2));
var flat = g2.lt(1e-8);
var g2s = g2.where(flat, 1);
var prof = p.pow(2).multiply(zxx).add(p.multiply(q).multiply(zxy).multiply(2)).add(q.pow(2).multiply(zyy))
            .divide(g2s).where(flat, 0);
var plan = q.pow(2).multiply(zxx).subtract(p.multiply(q).multiply(zxy).multiply(2)).add(p.pow(2).multiply(zyy))
            .divide(g2s).where(flat, 0);

var elev = to250(dem, ee.Reducer.mean());
var slp = to250(slope, ee.Reducer.mean());
var sinA = to250(aspect.multiply(Math.PI / 180).sin(), ee.Reducer.mean());
var cosA = to250(aspect.multiply(Math.PI / 180).cos(), ee.Reducer.mean());
var asp = sinA.atan2(cosA).multiply(180 / Math.PI).add(360).mod(360).where(slp.lt(2), -1);
var planA = to250(plan, ee.Reducer.mean());
var profA = to250(prof, ee.Reducer.mean());
var relief = to250(dem, ee.Reducer.max()).subtract(to250(dem, ee.Reducer.min()));

// ---------------- TWI (MERIT Hydro, 90 m)
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var mslope = ee.Terrain.slope(merit.select('elv'));
var twi = merit.select('upa').multiply(1e6).divide(ee.Image.pixelArea().sqrt())
          .divide(mslope.max(0.1).multiply(Math.PI / 180).tan()).log()
          .setDefaultProjection(merit.select('upa').projection());
var twiA = to250(twi, ee.Reducer.mean());

// ---------------- rainfall (CHIRPS pentad 1991-2020)
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD').filterDate('1991-01-01', '2021-01-01').select('precipitation');
var annual = chirps.sum().divide(30);
var months = [];
for (var m = 1; m <= 12; m++) months.push(chirps.filter(ee.Filter.calendarRange(m, m, 'month')).sum().divide(30));
var quarters = [];
for (var i = 0; i < 12; i++) quarters.push(months[i].add(months[(i + 1) % 12]).add(months[(i + 2) % 12]));
var wetq = ee.ImageCollection(quarters).max();
var cp = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD').first().projection();
var rainA = annual.setDefaultProjection(cp).resample('bilinear').reproject({crs: CRS, crsTransform: T250});
var rainQ = wetq.setDefaultProjection(cp).resample('bilinear').reproject({crs: CRS, crsTransform: T250});

// ---------------- land cover (WorldCover 2021, mode) and oblast id (mode)
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var lulc = to250(wc.reproject({crs: CRS, crsTransform: T50}), ee.Reducer.mode());
var oblast = to250(adm.reduceToImage({properties: ['oblast_id'], reducer: ee.Reducer.first()})
                      .reproject({crs: CRS, crsTransform: T50}), ee.Reducer.mode());

// ---------------- distances
var grip = ee.FeatureCollection('projects/sat-io/open-datasets/GRIP4/Middle-East-Central-Asia').filterBounds(AOI_LL);
var dRoad = to250(grip.distance({searchRadius: 32000, maxError: 25}).reproject({crs: CRS, crsTransform: T50}),
                  ee.Reducer.min()).unmask(32000).min(32000);
var chan = merit.select('upa').gte(10).reproject({crs: CRS, crsTransform: T100});
var dRiv = chan.fastDistanceTransform(256).sqrt().multiply(100).reproject({crs: CRS, crsTransform: T100});
var dRivA = to250(dRiv, ee.Reducer.mean()).min(25600);

// ---------------- NDVI (MODIS MOD13Q1, Jun-Aug median 2019-2023) and SoilGrids topsoil
var mod = ee.ImageCollection('MODIS/061/MOD13Q1');
var ndvi = mod.filterDate('2019-01-01', '2024-01-01').filter(ee.Filter.calendarRange(6, 8, 'month'))
              .select('NDVI').median().setDefaultProjection(mod.first().select('NDVI').projection());
function top30(v) {
  var im = ee.Image('projects/soilgrids-isric/' + v + '_mean');
  return im.select(v + '_0-5cm_mean').multiply(5).add(im.select(v + '_5-15cm_mean').multiply(10))
           .add(im.select(v + '_15-30cm_mean').multiply(15)).divide(30).setDefaultProjection(im.select(0).projection());
}

function i16(img, names) { return img.rename(names).unmask(NODATA).toInt16(); }
var parts = {
  kg250_terrainA: i16(ee.Image.cat([elev.round(), slp.multiply(10).round(), asp.round()]), ['elevation', 'slope', 'aspect']),
  kg250_terrainB: i16(ee.Image.cat([planA.multiply(1e5).round().clamp(-32000, 32000), profA.multiply(1e5).round().clamp(-32000, 32000)]), ['plan_curv', 'profile_curv']),
  kg250_terrainC: i16(ee.Image.cat([relief.round(), twiA.multiply(100).round()]), ['relief', 'twi']),
  kg250_envA: i16(ee.Image.cat([rainA.round(), rainQ.round(), lulc, oblast]), ['rain_annual', 'rain_wetq', 'lulc', 'oblast']),
  kg250_envB: i16(ee.Image.cat([dRoad.round(), dRivA.round()]), ['dist_roads', 'dist_rivers']),
  kg250_envC: i16(ee.Image.cat([to250(ndvi, ee.Reducer.mean()).round(), to250(top30('clay'), ee.Reducer.mean()).round(),
                                to250(top30('sand'), ee.Reducer.mean()).round()]), ['ndvi', 'clay', 'sand'])
};
Object.keys(parts).forEach(function (k) {
  Export.image.toDrive({image: parts[k], description: k, folder: FOLDER, fileNamePrefix: k, region: REGION,
    crs: CRS, crsTransform: T250, maxPixels: 1e10, fileFormat: 'GeoTIFF'});
});
Map.centerObject(adm, 7);
Map.addLayer(parts.kg250_terrainA.select('slope').divide(10).updateMask(parts.kg250_terrainA.select('slope').gte(0)),
             {min: 0, max: 40, palette: ['ffffff', 'f7e463', 'ef8a35', 'b5192c']}, 'slope 250 m');
print('6 export tasks queued (Tasks tab). Bands:', ee.Dictionary(Object.keys(parts).reduce(function (o, k) {
  o[k] = parts[k].bandNames(); return o; }, {})));
