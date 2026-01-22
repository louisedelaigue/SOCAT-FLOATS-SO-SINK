import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_obs-mob_glo_phy_my_0.125deg_P1M-m",
    variables=["so", "to", "mlotst"],
    minimum_longitude=-179.9375,
    maximum_longitude=179.9375,
    minimum_latitude=-82.1875,
    maximum_latitude=89.9375,
    minimum_depth=0,
    maximum_depth=0,
    output_directory="/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/ARMOR-3D"
)
