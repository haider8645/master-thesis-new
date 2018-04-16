import lmdb
lmdb_env = lmdb.open('/home/lod/master-thesis/LMDB-datasets/kipro/train_lmdb', readonly=True)
print lmdb_env.stat()
