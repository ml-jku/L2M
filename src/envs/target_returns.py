# max return in respective dataset * 2
MT50_targets = {
    'assembly-v1': 1450286.625, 'button-press-v1': 566043.703125, 'disassemble-v1': -9.405543645222982,
    'plate-slide-side-v1': 374616.09375, 'door-lock-v1': 544039.125, 'door-unlock-v1': 501917.859375,
    'pick-place-v1': 477427.6875, 'drawer-close-v1': 521100.375, 'coffee-pull-v1': 520720.3125,
    'handle-pull-v1': 493718.203125, 'peg-insert-side-v1': 307802.578125, 'door-close-v1': 578117.109375,
    'button-press-topdown-v1': 522984.65625, 'lever-pull-v1': -1.220013936360677, 'reach-v1': 638514.28125,
    'door-open-v1': 501558.234375, 'sweep-v1': 122089.7578125, 'box-close-v1': 589573.546875,
    'button-press-wall-v1': 537146.484375, 'basketball-v1': 500070.1875, 'soccer-v1': 514502.15625,
    'handle-press-v1': 493111.78125, 'coffee-button-v1': 552852.75, 'faucet-open-v1': 154400.8125,
    'plate-slide-v1': 339928.546875, 'reach-wall-v1': 626686.5, 'handle-pull-side-v1': 453395.90625,
    'drawer-open-v1': 488456.0625, 'dial-turn-v1': 450512.625, 'sweep-into-v1': 536743.265625,
    'bin-picking-v1': 687246.1875, 'coffee-push-v1': 534557.34375, 'pick-out-of-hole-v1': -11.327592213948568,
    'button-press-topdown-wall-v1': 522397.078125, 'plate-slide-back-side-v1': 530872.96875,
    'plate-slide-back-v1': 532280.8125, 'stick-push-v1': 178049.05078125, 'hand-insert-v1': 585300.9375,
    'window-open-v1': 356100.890625, 'pick-place-wall-v1': 506045.0625, 'hammer-v1': 324996.609375,
    'peg-unplug-side-v1': 549168.65625, 'window-close-v1': 293120.0390625, 'faucet-close-v1': 333695.90625,
    'handle-press-side-v1': 472267.875, 'push-back-v1': 519344.109375, 'push-v1': 539754.421875,
    'push-wall-v1': 522941.578125, 'stick-pull-v1': 1216821.9375, 'shelf-place-v1': 319610.765625
}

MT50_targets_v2 = {
    'assembly-v2': 1285.642, 'button-press-v2': 1604.5227, 'disassemble-v2': 1536.4567,
    'plate-slide-side-v2': 1748.304, 'door-lock-v2': 1831.5373, 'door-unlock-v2': 1794.7213,
    'pick-place-v2': 1300.692, 'drawer-close-v2': 1880.0, 'coffee-pull-v2': 1475.0727,
    'handle-pull-v2': 1759.07, 'peg-insert-side-v2': 1695.9867, 'door-close-v2': 1587.8,
    'button-press-topdown-v2': 1384.7, 'lever-pull-v2': 1677.1973, 'reach-v2': 1905.1067,
    'door-open-v2': 1608.812, 'sweep-v2': 1560.82, 'box-close-v2': 1127.9013,
    'button-press-wall-v2': 1614.572, 'basketball-v2': 1597.018, 'soccer-v2': 1706.2247,
    'handle-press-v2': 1929.8347, 'coffee-button-v2': 1698.32, 'faucet-open-v2': 1783.7487,
    'plate-slide-v2': 1713.0807, 'reach-wall-v2': 1863.1053, 'handle-pull-side-v2': 1710.8367,
    'drawer-open-v2': 1762.81, 'dial-turn-v2': 1887.328, 'sweep-into-v2': 1815.9427,
    'bin-picking-v2': 1308.598, 'coffee-push-v2': 1694.6847, 'pick-out-of-hole-v2': 1588.1173,
    'button-press-topdown-wall-v2': 1388.598, 'plate-slide-back-side-v2': 1833.7033,
    'plate-slide-back-v2': 1806.1567, 'stick-push-v2': 1629.2633, 'hand-insert-v2': 1789.0467,
    'window-open-v2': 1717.4207, 'pick-place-wall-v2': 1654.644, 'hammer-v2': 1724.842,
    'peg-unplug-side-v2': 1619.82, 'window-close-v2': 1559.3093, 'faucet-close-v2': 1798.588,
    'handle-press-side-v2': 1897.76, 'push-back-v2': 1564.7233, 'push-v2': 1791.2247,
    'push-wall-v2': 1746.7047, 'stick-pull-v2': 1551.368, 'shelf-place-v2': 1513.3307
}

ATARI_targets = {
    'AlienNoFrameskip-v4': 218.0, 'AmidarNoFrameskip-v4': 204.0, 'AssaultNoFrameskip-v4': 160.0,
    'AsterixNoFrameskip-v4': 104.0, 'AtlantisNoFrameskip-v4': 1444.0, 'BankHeistNoFrameskip-v4': 86.0, 
    'BattleZoneNoFrameskip-v4': 32.0, 'BeamRiderNoFrameskip-v4': 152.0, 'BoxingNoFrameskip-v4': 88.0, 
    'BreakoutNoFrameskip-v4': 95.0, 'CarnivalNoFrameskip-v4': 45.0, 'CentipedeNoFrameskip-v4': 137.0, 
    'ChopperCommandNoFrameskip-v4': 70.0, 'CrazyClimberNoFrameskip-v4': 735.0, 'DemonAttackNoFrameskip-v4': 231.0, 
    'DoubleDunkNoFrameskip-v4': 14.0, 'EnduroNoFrameskip-v4': 1091.0, 'FishingDerbyNoFrameskip-v4': 53.0,
    'FreewayNoFrameskip-v4': 34.0, 'FrostbiteNoFrameskip-v4': 71.0, 'GopherNoFrameskip-v4': 627.0, 
    'GravitarNoFrameskip-v4': 6.0, 'HeroNoFrameskip-v4': 201.0, 'IceHockeyNoFrameskip-v4': 2.0,
    'JamesbondNoFrameskip-v4': 17.0, 'KangarooNoFrameskip-v4': 74.0, 'KrullNoFrameskip-v4': 741.0,
    'KungFuMasterNoFrameskip-v4': 275.0, 'MsPacmanNoFrameskip-v4': 403.0, 'NameThisGameNoFrameskip-v4': 640.0, 
    'PhoenixNoFrameskip-v4': 67.0, 'PongNoFrameskip-v4': 21.0, 'PooyanNoFrameskip-v4': 427.0,
    'QbertNoFrameskip-v4': 588.0, 'RiverraidNoFrameskip-v4': 250.0, 'RoadRunnerNoFrameskip-v4': 217.0, 
    'RobotankNoFrameskip-v4': 73.0, 'SeaquestNoFrameskip-v4': 315.0, 'SpaceInvadersNoFrameskip-v4': 318.0, 
    'StarGunnerNoFrameskip-v4': 261.0, 'TimePilotNoFrameskip-v4': 24.0, 'UpNDownNoFrameskip-v4': 246.0, 
    'VideoPinballNoFrameskip-v4': 1078.0, 'WizardOfWorNoFrameskip-v4': 46.0, 
    'YarsRevengeNoFrameskip-v4': 207.0, 'ZaxxonNoFrameskip-v4': 30.0
}

# unfortunately DMControl produces these wird environment names
DMCONTROL_targets = {
    'acrobot-swingup': 158.764,
    'ball_in_cup-catch': 1000.0,
    'cartpole-balance': 993.194, 
    'cartpole-swingup': 857.492,
    'cheetah-run': 451.274,
    'finger-spin': 987.0, 
    'finger-turn_easy': 1000.0,
    'finger-turn_hard': 1000.0, 
    'fish-swim': 794.356, 
    'fish-upright': 999.433, 
    'hopper-hop': 116.777, 
    'hopper-stand': 975.575,
    'humanoid-run': 2.567, 
    'humanoid-stand': 17.218, 
    'humanoid-walk': 12.589, 
    'manipulator-bring_ball': 19.022, 
    'manipulator-insert_ball': 1000.0,
    'manipulator-insert_peg': 1000.0,
    'pendulum-swingup': 1000.0, 
    'point_mass-easy': 998.373,
    'reacher-easy': 1000.0, 
    'reacher-hard': 1000.0,
    'swimmer-swimmer15': 1000.0, 
    'swimmer-swimmer6': 1000.0,
    'walker-run': 513.464, 
    'walker-stand': 999.82,
    'walker-walk': 982.159
}

ALL_TARGETS = {**MT50_targets, **MT50_targets_v2, **ATARI_targets, **DMCONTROL_targets}
