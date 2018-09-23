import lmdb
lmdb_env = lmdb.open('/home/lod/datasets/trashnet/data/test_lmdb', readonly=True)
print lmdb_env.stat()
