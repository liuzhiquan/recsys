
DROP TABLE IF EXISTS `rec_torrents`;
CREATE TABLE `rec_torrents` (
  `torrentid` int(11) NOT NULL,
  `torrents` varchar(220) NOT NULL,
  PRIMARY KEY (`torrentid`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

