CREATE TABLE default.schema_migrations\n(\n    `version` Int64,\n    `dirty` UInt8,\n    `sequence` UInt64\n)\nENGINE = Distributed(\'{cluster}\', \'default\', \'schema_migrations_local\', rand())\n
