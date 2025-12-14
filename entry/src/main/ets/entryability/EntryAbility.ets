import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window, AppStorageV2 } from '@kit.ArkUI';
import { ActiveFormState, APP_STORAGE_KEY_ACTIVE_FORM, ROUTER_KEY_FORM_ID } from '../common/ActiveFormatState'



const DOMAIN = 0x0000;
const TAG: string = 'HarmonyStock';



export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    try {
      this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    } catch (err) {
      hilog.error(DOMAIN, TAG, 'Failed to set colorMode. Cause: %{public}s', JSON.stringify(err));
    }
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onCreate');

    // ✅ 冷启动：从 want 里取 formId，写入 AppStorageV2 全局对象
    this.consumeFormIdIntoGlobalState(want, 'onCreate');
  }

  /**
   * ✅ 热启动：当应用已在前台/后台，再次从桌面卡片点击进入，会走这里
   * 你日志里已经有 OnNewWant / onNewWant，因此这里必须也处理一次
   */
  onNewWant(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onNewWant');
    this.consumeFormIdIntoGlobalState(want, 'onNewWant');
  }

  onDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageCreate');

    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, TAG, 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, TAG, 'Succeeded in loading the content.');
    });
  }

  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onBackground');
  }

  // =========================================================
  // ✅ 核心：把桌面卡片 router 传入的 formId 写入 AppStorageV2
  // =========================================================
  private consumeFormIdIntoGlobalState(want: Want, from: string): void {
    const formId = this.readStringParamFromWant(want, ROUTER_KEY_FORM_ID)
    if (formId.length === 0) {
      return
    }

    const st = AppStorageV2.connect(
      ActiveFormState,
      APP_STORAGE_KEY_ACTIVE_FORM,
      () => new ActiveFormState()
    )

    if (st !== undefined) {
      st.formId = formId
      hilog.info(DOMAIN, TAG, '[%{public}s] saved formId=%{public}s into AppStorageV2', from, formId)
    } else {
      hilog.error(DOMAIN, TAG, 'AppStorageV2.connect failed')
    }
  }


  private getOrCreateActiveFormState(): ActiveFormState | null {
    const st: ActiveFormState | undefined =
      AppStorageV2.connect(
        ActiveFormState,
        APP_STORAGE_KEY_ACTIVE_FORM,
        () => new ActiveFormState()
      )

    if (!st) {
      hilog.error(DOMAIN, TAG, 'AppStorageV2.connect failed in Ability')
      return null
    }
    return st
  }

  // =========================================================
  // ✅ 工具函数：从 want.parameters 里安全读 string（无 any / unknown）
  // =========================================================
  private readStringParamFromWant(want: Want, key: string): string {
    if (!want.parameters) {
      return '';
    }

    const params: Record<string, Object> = want.parameters as Record<string, Object>;
    const v: Object | undefined = params[key];

    if (v === undefined || v === null) {
      return '';
    }
    if (typeof v === 'string') {
      return (v as string).trim();
    }
    return '';
  }
}
