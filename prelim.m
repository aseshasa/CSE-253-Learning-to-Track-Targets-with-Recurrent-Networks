clear all;
close all;

gt = [[ 1703.,   385.,   157.,   339.];
       [ 1699.,   383.,   159.,   341.];
       [ 1697.,   383.,   159.,   343.];
       [ 1695.,   383.,   159.,   343.];
       [ 1693.,   381.,   159.,   347.];
       [ 1689.,   381.,   161.,   349.];
       [ 1687.,   381.,   161.,   349.];
       [ 1685.,   379.,   161.,   353.];
       [ 1683.,   379.,   161.,   353.];
       [ 1679.,   379.,   163.,   355.];
       [ 1677.,   379.,   165.,   357.];
       [ 1675.,   377.,   165.,   359.];
       [ 1673.,   377.,   165.,   361.];
       [ 1669.,   377.,   167.,   361.];
       [ 1667.,   375.,   167.,   365.];
       [ 1665.,   375.,   167.,   367.];
       [ 1663.,   375.,   167.,   367.];
       [ 1659.,   375.,   169.,   369.];
       [ 1657.,   373.,   169.,   371.];
       [ 1655.,   373.,   169.,   373.];
       [ 1653.,   373.,   171.,   375.];
       [ 1649.,   371.,   173.,   377.];
       [ 1647.,   371.,   173.,   379.];
       [ 1645.,   371.,   173.,   379.];
       [ 1643.,   371.,   173.,   381.];
       [ 1639.,   369.,   175.,   385.];
       [ 1637.,   369.,   175.,   385.];
       [ 1635.,   369.,   175.,   387.];
       [ 1633.,   367.,   175.,   389.];
       [ 1629.,   367.,   179.,   391.];
       [ 1627.,   367.,   179.,   393.];
       [ 1625.,   365.,   179.,   395.];
       [ 1623.,   365.,   179.,   397.];
       [ 1619.,   365.,   181.,   397.];
       [ 1617.,   365.,   181.,   399.];
       [ 1615.,   363.,   181.,   403.];
       [ 1613.,   363.,   181.,   403.];
       [ 1609.,   363.,   183.,   405.];
       [ 1607.,   361.,   183.,   407.];
       [ 1605.,   361.,   185.,   409.];
       [ 1603.,   361.,   185.,   411.];
       [ 1599.,   361.,   187.,   411.];
       [ 1597.,   359.,   187.,   415.];
       [ 1595.,   359.,   187.,   415.];
       [ 1593.,   359.,   187.,   417.];
       [ 1589.,   357.,   189.,   421.];
       [ 1587.,   357.,   189.,   421.];
       [ 1585.,   357.,   189.,   423.];
       [ 1583.,   357.,   191.,   425.];
       [ 1579.,   355.,   191.,   429.];
       [ 1575.,   355.,   193.,   431.];
       [ 1571.,   353.,   193.,   435.];
       [ 1567.,   353.,   195.,   437.];
       [ 1563.,   353.,   195.,   439.];
       [ 1559.,   351.,   197.,   443.];
       [ 1555.,   351.,   197.,   447.];
       [ 1551.,   349.,   199.,   451.];
       [ 1547.,   349.,   199.,   453.];
       [ 1543.,   349.,   201.,   455.];
       [ 1539.,   347.,   201.,   459.];
       [ 1535.,   347.,   203.,   461.];
       [ 1531.,   345.,   203.,   465.];
       [ 1527.,   345.,   205.,   469.];
       [ 1523.,   345.,   205.,   471.];
       [ 1519.,   343.,   207.,   475.];
       [ 1515.,   343.,   207.,   477.];
       [ 1511.,   341.,   209.,   481.];
       [ 1507.,   341.,   209.,   483.];
       [ 1505.,   341.,   209.,   487.];
       [ 1499.,   339.,   211.,   489.];
       [ 1495.,   339.,   211.,   491.];
       [ 1491.,   337.,   211.,   493.];
       [ 1487.,   337.,   211.,   495.];
       [ 1481.,   335.,   213.,   499.];
       [ 1477.,   335.,   213.,   499.];
       [ 1473.,   333.,   215.,   503.];
       [ 1469.,   333.,   215.,   503.];
       [ 1463.,   331.,   217.,   507.];
       [ 1459.,   331.,   217.,   509.];
       [ 1455.,   329.,   217.,   511.];
       [ 1451.,   329.,   217.,   513.];
       [ 1445.,   327.,   221.,   515.];
       [ 1441.,   327.,   221.,   517.];
       [ 1437.,   325.,   221.,   521.];
       [ 1433.,   325.,   221.,   521.];
       [ 1427.,   323.,   223.,   525.];
       [ 1423.,   323.,   223.,   525.];
       [ 1419.,   321.,   225.,   529.];
       [ 1415.,   321.,   225.,   531.];
       [ 1409.,   319.,   227.,   533.];
       [ 1405.,   319.,   227.,   535.];
       [ 1401.,   317.,   227.,   537.];
       [ 1397.,   317.,   227.,   539.];
       [ 1393.,   317.,   229.,   541.];
       [ 1383.,   315.,   231.,   545.];
       [ 1375.,   313.,   233.,   551.];
       [ 1367.,   311.,   235.,   557.];
       [ 1359.,   309.,   237.,   563.];
       [ 1349.,   307.,   239.,   569.];
       [ 1341.,   307.,   241.,   573.]];
       
true_det = [[ 1689.  ,   385.  ,   146.62,   332.71];
       [ 1689.  ,   385.  ,   146.62,   332.71];
       [ 1664.  ,   418.  ,   135.42,   307.29];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1692.  ,   371.  ,   159.84,   362.7 ];
       [ 1666.  ,   385.  ,   146.62,   332.71];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1727.  ,   418.  ,   135.42,   307.29];
       [ 1689.  ,   385.  ,   146.62,   332.71];
       [ 1689.  ,   385.  ,   146.62,   332.71];
       [ 1689.  ,   408.  ,   146.62,   332.71];
       [ 1689.  ,   385.  ,   146.62,   332.71];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1666.  ,   385.  ,   146.62,   332.71];
       [ 1594.  ,   396.  ,   159.84,   362.7 ];
       [ 1594.  ,   396.  ,   159.84,   362.7 ];
       [ 1643.  ,   371.  ,   159.84,   362.7 ];
       [ 1594.  ,   396.  ,   159.84,   362.7 ];
       [ 1594.  ,   371.  ,   159.84,   362.7 ];
       [ 1589.  ,   381.  ,   175.68,   398.65];
       [ 1668.  ,   371.  ,   159.84,   362.7 ];
       [ 1619.  ,   396.  ,   159.84,   362.7 ];
       [ 1619.  ,   396.  ,   159.84,   362.7 ];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1619.  ,   371.  ,   159.84,   362.7 ];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1619.  ,   396.  ,   159.84,   362.7 ];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   354.  ,   175.68,   398.65];
       [ 1616.  ,   381.  ,   175.68,   398.65];
       [ 1622.  ,   354.  ,   189.32,   429.61];
       [ 1622.  ,   354.  ,   189.32,   429.61];
       [ 1622.  ,   354.  ,   189.32,   429.61];
       [ 1616.  ,   381.  ,   175.68,   398.65];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1622.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1564.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1528.  ,   322.  ,   207.45,   470.74];
       [ 1564.  ,   354.  ,   189.32,   429.61];
       [ 1564.  ,   354.  ,   189.32,   429.61];
       [ 1593.  ,   354.  ,   189.32,   429.61];
       [ 1528.  ,   353.  ,   207.45,   470.74];
       [ 1528.  ,   353.  ,   207.45,   470.74];
       [ 1702.  ,   283.  ,   226.74,   514.53];
       [ 1622.  ,   354.  ,   189.32,   429.61];
       [ 1528.  ,   322.  ,   207.45,   470.74];
       [ 1528.  ,   322.  ,   207.45,   470.74];
       [ 1702.  ,   283.  ,   226.74,   514.53];
       [ 1528.  ,   322.  ,   207.45,   470.74];
       [ 1535.  ,   408.  ,   175.68,   398.65];
       [ 1535.  ,   354.  ,   189.32,   429.61];
       [ 1528.  ,   353.  ,   207.45,   470.74];
       [ 1528.  ,   353.  ,   207.45,   470.74];
       [ 1496.  ,   353.  ,   207.45,   470.74];
       [ 1496.  ,   353.  ,   207.45,   470.74];
       [ 1496.  ,   353.  ,   207.45,   470.74];
       [ 1496.  ,   353.  ,   207.45,   470.74];
       [ 1528.  ,   353.  ,   207.45,   470.74];
       [ 1496.  ,   353.  ,   207.45,   470.74];
       [ 1492.  ,   318.  ,   226.74,   514.53];
       [ 1492.  ,   318.  ,   226.74,   514.53];
       [ 1492.  ,   318.  ,   226.74,   514.53];
       [ 1492.  ,   318.  ,   226.74,   514.53];
       [ 1496.  ,   353.  ,   207.45,   470.74];
       [ 1458.  ,   318.  ,   226.74,   514.53];
       [ 1458.  ,   318.  ,   226.74,   514.53];
       [ 1458.  ,   318.  ,   226.74,   514.53];
       [ 1464.  ,   353.  ,   207.45,   470.74];
       [ 1458.  ,   318.  ,   226.74,   514.53];
       [ 1458.  ,   318.  ,   226.74,   514.53];
       [ 1458.  ,   318.  ,   226.74,   514.53];
       [ 1432.  ,   353.  ,   207.45,   470.74];
       [ 1423.  ,   353.  ,   226.74,   514.53];
       [ 1388.  ,   318.  ,   226.74,   514.53];
       [ 1617.  ,   254.  ,   270.83,   614.58];
       [ 1432.  ,   385.  ,   207.45,   470.74];
       [ 1617.  ,   254.  ,   270.83,   614.58];
       [  984.  ,   290.  ,   207.45,   470.74];
       [ 1400.  ,   385.  ,   207.45,   470.74];
       [ 1400.  ,   385.  ,   207.45,   470.74];
       [ 1400.  ,   385.  ,   207.45,   470.74];
       [ 1400.  ,   417.  ,   207.45,   470.74];
       [ 1400.  ,   417.  ,   207.45,   470.74]];
       
gt2 = [[ 1293.,   455.,    83.,   213.];
       [ 1293.,   455.,    83.,   213.];
       [ 1293.,   455.,    83.,   213.];
       [ 1293.,   455.,    83.,   213.];
       [ 1295.,   455.,    83.,   213.];
       [ 1297.,   455.,    83.,   213.];
       [ 1299.,   455.,    83.,   213.];
       [ 1301.,   455.,    83.,   213.];
       [ 1303.,   455.,    83.,   213.];
       [ 1305.,   455.,    83.,   213.];
       [ 1307.,   455.,    83.,   213.];
       [ 1309.,   455.,    83.,   213.];
       [ 1313.,   455.,    83.,   215.];
       [ 1315.,   455.,    83.,   215.];
       [ 1317.,   455.,    83.,   215.];
       [ 1319.,   455.,    83.,   215.];
       [ 1321.,   455.,    83.,   215.];
       [ 1323.,   455.,    83.,   215.];
       [ 1325.,   455.,    83.,   215.];
       [ 1327.,   455.,    83.,   215.];
       [ 1331.,   457.,    83.,   215.];
       [    7.,   237.,   241.,   673.];
       [    5.,   237.,   249.,   673.];
       [    5.,   237.,   255.,   673.];
       [    5.,   237.,   261.,   673.];
       [    5.,   239.,   267.,   671.];
       [   15.,   239.,   275.,   669.];
       [   27.,   239.,   281.,   669.];
       [   39.,   239.,   287.,   669.];
       [   49.,   241.,   295.,   667.];
       [   61.,   241.,   301.,   667.];
       [   73.,   241.,   307.,   667.];
       [   83.,   243.,   315.,   665.];
       [   95.,   243.,   321.,   665.];
       [  107.,   243.,   327.,   665.];
       [  119.,   245.,   335.,   663.];
       [  137.,   245.,   327.,   659.];
       [  157.,   245.,   319.,   657.];
       [  175.,   245.,   313.,   653.];
       [  195.,   247.,   305.,   649.];
       [  213.,   247.,   299.,   647.];
       [  233.,   247.,   291.,   643.];
       [  251.,   249.,   285.,   639.];
       [  271.,   249.,   277.,   635.];
       [  289.,   249.,   271.,   633.];
       [  309.,   251.,   263.,   629.];
       [  327.,   251.,   257.,   625.];
       [  347.,   251.,   249.,   623.];
       [  367.,   253.,   241.,   619.];
       [  381.,   253.,   239.,   615.];
       [  395.,   255.,   239.,   611.];
       [  409.,   255.,   239.,   609.];
       [  425.,   257.,   237.,   605.];
       [  439.,   259.,   235.,   601.];
       [  453.,   259.,   235.,   597.];
       [  467.,   261.,   235.,   593.];
       [  483.,   263.,   233.,   589.];
       [  497.,   263.,   231.,   587.];
       [  511.,   265.,   231.,   583.];
       [  525.,   265.,   231.,   581.];
       [  541.,   267.,   229.,   575.];
       [  555.,   269.,   229.,   571.];
       [  569.,   269.,   227.,   569.];
       [  583.,   271.,   227.,   565.];
       [  599.,   273.,   225.,   561.];
       [  613.,   273.,   225.,   559.];
       [  627.,   275.,   223.,   553.];
       [  641.,   275.,   223.,   551.];
       [  657.,   277.,   221.,   547.];
       [  671.,   279.,   221.,   543.];
       [  685.,   279.,   221.,   541.];
       [  699.,   281.,   219.,   535.];
       [  715.,   283.,   217.,   531.];
       [  729.,   283.,   217.,   529.];
       [  743.,   285.,   217.,   525.];
       [  757.,   285.,   215.,   523.];
       [  773.,   287.,   213.,   519.];
       [  787.,   289.,   213.,   513.];
       [  801.,   289.,   213.,   511.];
       [  815.,   291.,   213.,   507.];
       [  831.,   293.,   209.,   503.];
       [  845.,   293.,   209.,   501.];
       [  859.,   295.,   209.,   497.];
       [  873.,   295.,   209.,   493.];
       [  889.,   297.,   205.,   489.];
       [  903.,   299.,   205.,   485.];
       [  917.,   299.,   205.,   483.];
       [  931.,   301.,   205.,   479.];
       [  947.,   303.,   203.,   475.];
       [  955.,   303.,   201.,   471.];
       [  965.,   305.,   199.,   467.];
       [  975.,   305.,   197.,   465.];
       [  985.,   307.,   195.,   461.];
       [  993.,   307.,   195.,   459.];
       [ 1003.,   309.,   193.,   455.];
       [ 1013.,   309.,   191.,   453.];
       [ 1023.,   311.,   189.,   449.];
       [ 1031.,   311.,   189.,   447.];
       [ 1041.,   313.,   187.,   443.];
       [ 1051.,   313.,   183.,   441.]];
       
true_det2 = [[ 1800.   ,   483.   ,    94.66 ,   214.81 ];
       [ 1312.   ,   503.   ,    61.514,   139.59 ];
       [ 1234.   ,   415.   ,   103.17 ,   234.13 ];
       [ 1668.   ,   371.   ,   159.84 ,   362.7  ];
       [ 1318.   ,   537.   ,    52.   ,   118.   ];
       [ 1741.   ,   473.   ,   113.37 ,   257.27 ];
       [ 1668.   ,   371.   ,   159.84 ,   362.7  ];
       [  257.   ,   455.   ,   113.37 ,   257.27 ];
       [ 1274.   ,   551.   ,    43.624,    98.993];
       [ 1278.   ,   545.   ,    47.794,   108.46 ];
       [ 1689.   ,   385.   ,   146.62 ,   332.71 ];
       [ 1619.   ,   396.   ,   159.84 ,   362.7  ];
       [ 1286.   ,   545.   ,    47.794,   108.46 ];
       [ 1619.   ,   396.   ,   159.84 ,   362.7  ];
       [ 1619.   ,   396.   ,   159.84 ,   362.7  ];
       [ 1619.   ,   396.   ,   159.84 ,   362.7  ];
       [ 1336.   ,   500.   ,    67.474,   153.11 ];
       [ 1295.   ,   461.   ,    87.838,   199.32 ];
       [ 1291.   ,   454.   ,    94.66 ,   214.81 ];
       [ 1308.   ,   474.   ,    87.838,   199.32 ];
       [ 1692.   ,   396.   ,   159.84 ,   362.7  ];
       [ 1314.   ,   468.   ,    79.918,   181.35 ];
       [  -52.   ,   254.   ,   270.83 ,   614.58 ];
       [  -52.   ,   254.   ,   270.83 ,   614.58 ];
       [  -10.   ,   254.   ,   270.83 ,   614.58 ];
       [ 1668.   ,   371.   ,   159.84 ,   362.7  ];
       [ 1562.   ,   381.   ,   175.68 ,   398.65 ];
       [   31.   ,   254.   ,   270.83 ,   614.58 ];
       [   31.   ,   254.   ,   270.83 ,   614.58 ];
       [ 1821.   ,   526.   ,   103.17 ,   234.13 ];
       [   31.   ,   254.   ,   270.83 ,   614.58 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [ 1295.   ,   461.   ,    87.838,   199.32 ];
       [ 1692.   ,   396.   ,   159.84 ,   362.7  ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [ 1310.   ,   545.   ,    52.   ,   118.   ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [  100.   ,   490.   ,   113.37 ,   257.27 ];
       [  329.   ,   543.   ,   135.42 ,   307.29 ];
       [  333.   ,   543.   ,   146.62 ,   332.71 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [ 1310.   ,   537.   ,    52.   ,   118.   ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [ 1315.   ,   545.   ,    47.794,   108.46 ];
       [  512.   ,   494.   ,    33.737,    76.557];
       [ 1670.   ,   381.   ,   175.68 ,   398.65 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [  336.   ,   236.   ,   250.   ,   567.31 ];
       [ 1315.   ,   545.   ,    47.794,   108.46 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [ 1315.   ,   545.   ,    47.794,   108.46 ];
       [ 1651.   ,   383.   ,   189.32 ,   429.61 ];
       [ 1651.   ,   383.   ,   189.32 ,   429.61 ];
       [   13.   ,   490.   ,   113.37 ,   257.27 ];
       [ 1643.   ,   408.   ,   175.68 ,   398.65 ];
       [ 1622.   ,   383.   ,   189.32 ,   429.61 ];
       [ 1535.   ,   325.   ,   189.32 ,   429.61 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [  532.   ,   213.   ,   270.83 ,   614.58 ];
       [ 1315.   ,   545.   ,    47.794,   108.46 ];
       [  566.   ,   236.   ,   250.   ,   567.31 ];
       [ 1592.   ,   353.   ,   207.45 ,   470.74 ];
       [ 1592.   ,   353.   ,   207.45 ,   470.74 ];
       [ 1622.   ,   383.   ,   189.32 ,   429.61 ];
       [ 1378.   ,   510.   ,    67.474,   153.11 ];
       [ 1308.   ,   461.   ,    87.838,   199.32 ];
       [  963.   ,   547.   ,    73.864,   167.61 ];
       [  689.   ,   283.   ,   226.74 ,   514.53 ];
       [  963.   ,   547.   ,    73.864,   167.61 ];
       [ 1560.   ,   322.   ,   207.45 ,   470.74 ];
       [  720.   ,   266.   ,   189.32 ,   429.61 ];
       [  728.   ,   258.   ,   207.45 ,   470.74 ];
       [  749.   ,   266.   ,   189.32 ,   429.61 ];
       [ 1464.   ,   322.   ,   207.45 ,   470.74 ];
       [  759.   ,   283.   ,   226.74 ,   514.53 ];
       [  409.   ,   447.   ,   103.17 ,   234.13 ];
       [ 1562.   ,   283.   ,   226.74 ,   514.53 ];
       [  424.   ,   447.   ,   103.17 ,   234.13 ];
       [ 1378.   ,   510.   ,    67.474,   153.11 ];
       [  424.   ,   447.   ,   103.17 ,   234.13 ];
       [ 1310.   ,   545.   ,    52.   ,   118.   ];
       [ 1458.   ,   318.   ,   226.74 ,   514.53 ];
       [ 1372.   ,   490.   ,    73.864,   167.61 ];
       [  888.   ,   290.   ,   207.45 ,   470.74 ];
       [  920.   ,   290.   ,   207.45 ,   470.74 ];
       [  920.   ,   290.   ,   207.45 ,   470.74 ];
       [  920.   ,   290.   ,   207.45 ,   470.74 ];
       [  920.   ,   290.   ,   207.45 ,   470.74 ];
       [ 1423.   ,   353.   ,   226.74 ,   514.53 ];
       [  952.   ,   290.   ,   207.45 ,   470.74 ];
       [  952.   ,   290.   ,   207.45 ,   470.74 ];
       [  984.   ,   290.   ,   207.45 ,   470.74 ];
       [  984.   ,   290.   ,   207.45 ,   470.74 ];
       [  984.   ,   290.   ,   207.45 ,   470.74 ];
       [ 1011.   ,   325.   ,   189.32 ,   429.61 ];
       [ 1028.   ,   322.   ,   159.84 ,   362.7  ];
       [ 1048.   ,   300.   ,   175.68 ,   398.65 ];
       [ 1400.   ,   417.   ,   207.45 ,   470.74 ];
       [ 1400.   ,   417.   ,   207.45 ,   470.74 ]];


imglist = dir('img2/*.jpg');
figure; hold on;
writerObj = VideoWriter('prelim3', 'MPEG-4');
open(writerObj);
for i = 1:100
	img = imread(['img2/', imglist(i).name]);
	imshow(img);
	rectangle('Position', gt(i,:), 'EdgeColor', [0 1 0], 'LineWidth', 2.5);
	rectangle('Position', true_det(i,:), 'EdgeColor', [1 0 0], 'LineWidth', 2.5);
	rectangle('Position', gt2(i,:), 'EdgeColor', [0 1 0], 'LineWidth', 2.5);
	rectangle('Position', true_det2(i,:), 'EdgeColor', [1 0 0], 'LineWidth', 2.5);
	f = getframe(gcf);
	vidf = frame2im(f);
	writeVideo(writerObj,vidf);
end
close(writerObj);