import lmdb
lmdb_env = lmdb.open('/home/lod/master-thesis/LMDB-datasets/kipro/validation_lmdb', readonly=True)
print lmdb_env.stat()
