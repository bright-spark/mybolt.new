import { WebContainer } from '@webcontainer/api';
import { atom, map, type MapStore } from 'nanostores';
import * as nodePath from 'node:path';
import type { BoltAction, FileAction, ShellAction, StartAction } from '~/types/actions';
import { createScopedLogger } from '~/utils/logger';
import { unreachable } from '~/utils/unreachable';
import type { ActionCallbackData } from './message-parser';
import type { BoltShell } from '~/utils/shell';

const logger = createScopedLogger('ActionRunner');

const EXECUTION_TIMEOUT = 30000; // 30 seconds
const START_ACTION_TIMEOUT = 2000; // 2 seconds
const MAX_RETRIES = 3;

export type ActionStatus = 'pending' | 'running' | 'complete' | 'aborted' | 'failed';

export type BaseActionState = BoltAction & {
  status: Exclude<ActionStatus, 'failed'>;
  abort: () => void;
  executed: boolean;
  abortSignal: AbortSignal;
  retryCount?: number;
  startTime?: number;
  endTime?: number;
};

export type FailedActionState = BoltAction &
  Omit<BaseActionState, 'status'> & {
    status: Extract<ActionStatus, 'failed'>;
    error: string;
  };

export type ActionState = BaseActionState | FailedActionState;

type BaseActionUpdate = Partial<
  Pick<BaseActionState, 'status' | 'abort' | 'executed' | 'retryCount' | 'startTime' | 'endTime'>
>;

export type ActionStateUpdate =
  | BaseActionUpdate
  | (Omit<BaseActionUpdate, 'status'> & { status: 'failed'; error: string });

type ActionsMap = MapStore<Record<string, ActionState>>;

export class ActionRunner {
  #webcontainer: Promise<WebContainer>;
  #currentExecutionPromise: Promise<void> = Promise.resolve();
  #shellTerminal: () => BoltShell;
  runnerId = atom<string>(`${Date.now()}`);
  actions: ActionsMap = map({});

  constructor(webcontainerPromise: Promise<WebContainer>, getShellTerminal: () => BoltShell) {
    this.#webcontainer = webcontainerPromise;
    this.#shellTerminal = getShellTerminal;
  }

  addAction(data: ActionCallbackData) {
    const { actionId } = data;
    logger.debug(`Adding action ${actionId}`);

    const actions = this.actions.get();
    const action = actions[actionId];

    if (action) {
      logger.debug(`Action ${actionId} already exists`);
      return;
    }

    const abortController = new AbortController();

    this.actions.setKey(actionId, {
      ...data.action,
      status: 'pending',
      executed: false,
      abort: () => {
        logger.debug(`Aborting action ${actionId}`);
        abortController.abort();
        this.#updateAction(actionId, {
          status: 'aborted',
          endTime: Date.now(),
        });
      },
      abortSignal: abortController.signal,
      startTime: Date.now(),
    });

    this.#currentExecutionPromise.then(() => {
      this.#updateAction(actionId, { status: 'running' });
    });
  }

  async runAction(data: ActionCallbackData, isStreaming: boolean = false) {
    const { actionId } = data;
    logger.debug(`Running action ${actionId}, streaming: ${isStreaming}`);

    const action = this.actions.get()[actionId];

    if (!action) {
      const error = `Action ${actionId} not found`;
      logger.error(error);
      unreachable(error);
    }

    if (action.executed) {
      logger.debug(`Action ${actionId} already executed, skipping`);
      return;
    }

    if (isStreaming && action.type !== 'file') {
      logger.debug(`Streaming not supported for action type ${action.type}`);
      return;
    }

    this.#updateAction(actionId, {
      ...action,
      ...data.action,
      executed: !isStreaming,
      startTime: Date.now(),
    });

    let isCompleted = false;
    let timeoutId: NodeJS.Timeout;

    const timeoutPromise = new Promise<void>((_, reject) => {
      timeoutId = setTimeout(() => {
        if (!isCompleted) {
          reject(new Error(`Action ${actionId} timed out after ${EXECUTION_TIMEOUT}ms`));
        }
      }, EXECUTION_TIMEOUT);
    });

    this.#currentExecutionPromise = Promise.race([
      this.#currentExecutionPromise
        .then(async () => {
          logger.debug(`Executing action ${actionId}`);
          await this.#executeAction(actionId, isStreaming);
          isCompleted = true;
          clearTimeout(timeoutId);
        })
        .catch((error) => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`Action ${actionId} failed: ${errorMessage}`, error);

          const action = this.actions.get()[actionId];
          const retryCount = (action.retryCount || 0) + 1;

          if (retryCount <= MAX_RETRIES && !action.abortSignal.aborted) {
            logger.debug(`Retrying action ${actionId} (attempt ${retryCount}/${MAX_RETRIES})`);
            this.#updateAction(actionId, {
              status: 'pending',
              retryCount,
              executed: false,
            });

            return this.runAction(data, isStreaming);
          }

          this.#updateAction(actionId, {
            status: 'failed',
            error: `Action failed: ${errorMessage}`,
            endTime: Date.now(),
          });

          return Promise.resolve();
        }),
      timeoutPromise.catch((error) => {
        if (!isCompleted) {
          logger.error(`Timeout error for action ${actionId}: ${error.message}`);
          this.#updateAction(actionId, {
            status: 'failed',
            error: `Timeout: ${error.message}`,
            endTime: Date.now(),
          });
        }

        return Promise.resolve();
      }),
    ]);
  }

  async #executeAction(actionId: string, isStreaming: boolean = false) {
    const action = this.actions.get()[actionId];
    logger.debug(`Executing action ${actionId} of type ${action.type}`);

    this.#updateAction(actionId, { status: 'running' });

    try {
      switch (action.type) {
        case 'shell': {
          await this.#runShellAction(action as ShellAction & ActionState);
          break;
        }
        case 'file': {
          await this.#runFileAction(action as FileAction & ActionState);
          break;
        }
        case 'start': {
          await this.#runStartAction(action as StartAction & ActionState);
          return;
        }
        default: {
          unreachable(`Unknown action type: ${(action as { type: string }).type}`);
        }
      }

      if (action.abortSignal.aborted) {
        logger.debug(`Action ${actionId} was aborted`);
        this.#updateAction(actionId, {
          status: 'aborted',
          endTime: Date.now(),
        });

        return;
      }

      const finalStatus = isStreaming ? 'running' : 'complete';
      logger.debug(`Action ${actionId} completed with status: ${finalStatus}`);

      this.#updateAction(actionId, {
        status: finalStatus,
        endTime: finalStatus === 'complete' ? Date.now() : undefined,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Action ${actionId} of type ${action.type} failed: ${errorMessage}`, error);

      this.#updateAction(actionId, {
        status: 'failed',
        error: `Action failed: ${errorMessage}`,
        endTime: Date.now(),
      });

      throw error;
    }
  }

  async #runShellAction(action: ShellAction & ActionState) {
    logger.debug(`Running shell action: ${action.content}`);

    const shell = await this.#getShell();
    const resp = await shell.executeCommand(this.runnerId.get(), action.content);

    logger.debug(`Shell command completed with exit code: ${resp?.exitCode}`);

    if (resp?.exitCode !== 0) {
      throw new Error(`Shell command failed with exit code ${resp?.exitCode}`);
    }
  }

  async #runStartAction(action: StartAction & ActionState) {
    logger.debug(`Running start action: ${action.content}`);

    const shell = await this.#getShell();

    // Create a promise that tracks the start action completion
    const startPromise = shell.executeCommand(this.runnerId.get(), action.content).then((resp) => {
      if (resp?.exitCode !== 0) {
        throw new Error(`Start command failed with exit code ${resp?.exitCode}`);
      }

      logger.debug('Start command completed successfully');
    });

    // Wait for either the command to complete or the timeout
    await Promise.race([startPromise, new Promise((resolve) => setTimeout(resolve, START_ACTION_TIMEOUT))]);
  }

  async #runFileAction(action: FileAction & ActionState) {
    const webcontainer = await this.#webcontainer;
    const folder = this.#normalizePath(action.filePath);

    logger.debug(`Writing file ${action.filePath}`);
    await this.#ensureDirectory(webcontainer, folder);
    await this.#writeAndVerifyFile(webcontainer, action);
  }

  async #getShell(): Promise<BoltShell> {
    const shell = this.#shellTerminal();
    await shell.ready();

    if (!shell?.terminal || !shell?.process) {
      throw new Error('Shell terminal not initialized');
    }

    return shell;
  }

  #normalizePath(filePath: string): string {
    const folder = nodePath.dirname(filePath);
    return folder.replace(/\/+$/g, '');
  }

  async #ensureDirectory(webcontainer: WebContainer, folder: string): Promise<void> {
    if (folder === '.') {
      return;
    }

    try {
      await webcontainer.fs.mkdir(folder, { recursive: true });
      logger.debug(`Created folder ${folder}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to create folder ${folder}: ${errorMessage}`, error);
      throw new Error(`Failed to create directory ${folder}: ${errorMessage}`);
    }
  }

  async #writeAndVerifyFile(webcontainer: WebContainer, action: FileAction & ActionState): Promise<void> {
    try {
      // Clean up existing file
      try {
        await webcontainer.fs.rm(action.filePath, { force: true });
        logger.debug(`Removed existing file ${action.filePath}`);
      } catch {
        logger.debug(`No existing file to remove at ${action.filePath}`);
      }

      // Write new file
      await webcontainer.fs.writeFile(action.filePath, action.content);
      logger.debug(`File written ${action.filePath}`);

      // Verify file contents
      const written = await webcontainer.fs.readFile(action.filePath, 'utf-8');

      if (written !== action.content) {
        throw new Error('File content verification failed - content mismatch');
      }

      logger.debug(`File content verified ${action.filePath}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to write/verify file ${action.filePath}: ${errorMessage}`, error);
      throw new Error(`File operation failed: ${errorMessage}`);
    }
  }

  #updateAction(id: string, newState: ActionStateUpdate) {
    const actions = this.actions.get();
    const currentAction = actions[id];

    // Preserve timing information
    const startTime = newState.status === 'pending' ? Date.now() : currentAction.startTime;
    const endTime = ['complete', 'failed', 'aborted'].includes(newState.status as string)
      ? Date.now()
      : newState.endTime;

    this.actions.setKey(id, {
      ...currentAction,
      ...newState,
      startTime,
      endTime,
    });

    logger.debug(`Updated action ${id} status to ${newState.status}`);
  }
}
