import copernicusmarine

# FULL DATASET
copernicusmarine.subset(
    dataset_id="cmems_obs-mob_glo_phy_my_0.125deg_P1M-m",
    variables=["so", "to", "mlotst"],
    minimum_longitude=-179.9375,
    maximum_longitude=179.9375,
    minimum_latitude=-82.1875,
    maximum_latitude=-20,
    minimum_depth=0,
    maximum_depth=0,
    start_datetime="2003-01-01T00:00:00",
    end_datetime="2024-12-31T23:59:59",
    output_directory="/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/ARMOR-3D"
)


# FEBRUARY 2023 SUBSET
# import copernicusmarine

# copernicusmarine.subset(
#     dataset_id="cmems_obs-mob_glo_phy_my_0.125deg_P1M-m",
#     variables=["so", "to", "mlotst"],
#     minimum_longitude=-179.9375,
#     maximum_longitude=179.9375,
#     minimum_latitude=-82.1875,
#     maximum_latitude=89.9375,
#     minimum_depth=0,
#     maximum_depth=0,
#     start_datetime="2023-02-01T00:00:00",
#     end_datetime="2023-02-28T23:59:59",
#     output_directory="/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/ARMOR-3D"
# )