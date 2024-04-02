import asyncio
import traceback

import aiofiles
import fsspec
import grpc
from fsspec.asyn import AsyncFileSystem

from unionai.filesystems._async_utils import mirror_sync_methods
from unionai.filesystems._endpoint import _create_channel
from unionai.internal.common import list_pb2
from unionai.internal.objectstore.definition_pb2 import Key, Metadata, Object
from unionai.internal.objectstore.metadata_service_pb2_grpc import MetadataStoreServiceStub
from unionai.internal.objectstore.payload_pb2 import (
    DeleteRequest,
    GetRequest,
    GetResponse,
    HeadRequest,
    HeadResponse,
    ListRequest,
    ListResponse,
    PutRequest,
)


class AsyncUnionMetaFS(AsyncFileSystem):
    mirror_sync_methods = False

    def _prefix_path(self, path):
        if path.startswith(f"{self.protocol}://"):
            return path
        return f"{self.protocol}://{path}"

    def __init__(self, *args, **kwargs):
        channel = _create_channel()
        self._client = MetadataStoreServiceStub(channel)
        loop = channel._loop
        asyncio.set_event_loop(loop)
        asyncio.events.set_event_loop(loop)
        super().__init__(loop=loop, *args, **kwargs)
        self._loop = loop
        self._metadata = None
        self.blocksize = 2**18  # 2MB
        mirror_sync_methods(self)

    async def _rm_file(self, path, **kwargs):
        path = self._prefix_path(path)
        print(f"Deleting {path}")
        await self._client.Delete(DeleteRequest(key=Key(key=path)))

    async def _cp_file(self, path1, path2, **kwargs):
        # TODO: Implement cp in remote metadata store
        _, temp_filename = await aiofiles.tempfile.TemporaryFile(suffix=".yaml")
        await self._get_file(path1, temp_filename, **kwargs)
        await self._put_file(temp_filename, path2, **kwargs)

    async def _pipe_file(self, path, value, **kwargs):
        path = self._prefix_path(path)
        req = PutRequest(metadata=Metadata(), key=Key(key=path), object=Object(contents=value))
        try:
            await self._client.Put(req)
        except Exception as e:
            traceback.print_exception(e)
            raise

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        path = self._prefix_path(path)
        req = GetRequest(key=Key(key=path))
        try:
            res: GetResponse = await self._client.Get(req)
            return res.object.contents
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise FileNotFoundError(path)
            traceback.print_exception(e)
            raise e
        except Exception as e:
            traceback.print_exception(e)
            raise

    async def _put_file(self, lpath, rpath, **kwargs):
        rpath = self._prefix_path(rpath)
        async with aiofiles.open(lpath, "rb") as f:
            await self._pipe_file(rpath, await f.read(), **kwargs)

    async def _get_file(self, rpath, lpath, **kwargs):
        rpath = self._prefix_path(rpath)
        async with aiofiles.open(lpath, "wb") as f:
            await f.write(await self._cat_file(rpath, **kwargs))

    async def _info(self, path, **kwargs):
        print(f"Info: {path}")
        path = self._prefix_path(path)
        req = HeadRequest(key=Key(key=path))
        try:
            res: HeadResponse = await self._client.Head(req)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise FileNotFoundError(path)
            traceback.print_exception(e)
            raise e
        except Exception as e:
            traceback.print_exception(e)
            raise

        return {
            "size": res.size_bytes,
            "etag": res.etag,
            "tags": res.metadata.tag if res.metadata and res.metadata.tag else {},
            "type": "file",
        }

    async def _ls(self, path, detail=True, **kwargs):
        path = self._prefix_path(path)
        if not path.endswith("*"):
            path = f"{path}*"

        req = ListRequest(
            request=list_pb2.ListRequest(
                limit=1000,
                filters=[
                    list_pb2.Filter(
                        field="prefix",
                        function=list_pb2.Filter.GREATER_THAN_OR_EQUAL,
                        values=[path],
                    )
                ],
            )
        )

        keys = []
        while True:
            res: ListResponse = await self._client.List(req)
            keys.extend(res.keys)
            if not res.next_token or res.next_token == "":
                break

        return [
            {
                "name": k,
                "type": "file",
            }
            for k in keys
        ]


fsspec.register_implementation(name="unionmeta", cls=AsyncUnionMetaFS, clobber=True)
