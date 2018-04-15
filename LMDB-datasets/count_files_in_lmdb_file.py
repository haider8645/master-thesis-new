import lmdb
lmdb_env = lmdb.open('/home/lod/master-thesis/LMDB-datasets/cherry_lmdb', readonly=True)
print lmdb_env.stat()
