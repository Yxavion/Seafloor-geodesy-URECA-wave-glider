%% Mapping of wave glider positions
% Yxavion Mar 2024
% Run this every time a new shape is produced and a map is needed. 

%% Loading in data of the gliders

% the first 3 are the transponder locations and the fourth row is the 
array_pos = [44.842325200, -125.134820280;
             44.817929650, -125.126649450;
             44.832681360, -125.099794900;
              44.8319, -125.1204];
% read the lat long data
wg_lat_log = readtable('lat_long_shape.csv');

%% plots

figure
geoplot(wg_lat_log, "lat", "long", "Marker",".", "LineStyle", "none", ...
    "MarkerSize", 15);
geobasemap landcover


